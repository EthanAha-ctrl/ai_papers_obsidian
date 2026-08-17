---
source_pdf: Code-as-Room Generating 3D Rooms from.pdf
paper_sha256: 3cf9bfda35ba662bee2c9b18ea774ba84ac44a20c40813ed88129312d88889c4
processed_at: '2026-08-03T16:25:26-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Code-as-Room

## 1. 一句话先抓重点

你给一张**从天花板往下看的房间俯视图**（floor plan那种），它吐出一段**Blender代码**，跑一下就是一个能编辑、能渲染、能走进去的3D房间。

就这么个事儿。但怎么把这事做稳，是这篇paper的全部艺术。

## 2. 为什么这事难？先吐槽一下前人

**用文字描述生成3D房间的方法**（Holodeck [30]、LayoutGPT [6] 这类）有个先天毛病：你说"床靠窗放"，模型根本不知道"靠"是10厘米还是1米，更不知道窗在哪面墙。文字天生表达不了精确空间信息。

**用图片生成3D的方法**里，VIGA [32] 是最接近的——它让VLM写Blender代码，然后render出来跟input图比对，循环修正。听着很美，但在top-down view上**直接翻车**：要么卡进infinite loop改不出来，要么fine-grained的spatial detail全丢。

为什么VIGA会翻车？我给你打个比方。VIGA让一个聪明小孩一口气把整个房间的代码写完，写错了自己检查自己改。问题是：
- 这小孩写着写着就忘了前面写过啥（context forgetting）
- 他自己critique自己的时候会说"要不把墙挪一下？"——但墙是input image定死的，挪墙就违反input了，于是又改回去，再critique又说要挪墙……**infinite loop**
- top-down view里物体多、小，他一口气处理不过来，**小物件直接漏掉**

CaR就是来修这三个毛病的。

## 3. CaR的pipeline：拆、记、卡死、修

整个pipeline 10个stage，我把它分成4个大块讲。intuition是：**把一个又大又模糊的活儿，拆成一串又小又明确的小活儿，每个小活儿都在VLM能力sweet spot里。**

### 3.1 Coarse Stage（Stage 1-2）：先把图看明白

**Stage 1** 就是让VLM看图，按schema吐出一段结构化描述 $D_1$。每个物体标注identifier、category、placement type（floor/wall/surface）、parent（如果它依附于别的东西）。walls/doors/windows被标记为**fixed spatial references**——后面谁都不许动它们。

这里有个trick叫**perimeter-aware prompt**：专门扫一遍墙、角、开口、粗网格，把贴墙、靠角的物体单独捞出来。不然VLM看top-down图很容易漏掉wall-mounted的东西（挂画、壁灯、窗户）。

**Stage 2** 在 $D_1$ 基础上搭一个**scene graph** $G=(V,E)$。关键的design choice是：**先确定骨架，再让VLM填空**。

$$S = \{V_{\text{arch}}, V_{\text{major}}, E_{\text{parent}}, M_{\text{minor}}\}$$

- $V_{\text{arch}}$：建筑元素节点（墙门窗）
- $V_{\text{major}}$：主要家具节点
- $E_{\text{parent}}$：从hierarchy推出的父子边（桌上的杯子parent是桌子）
- $M_{\text{minor}}$：小物件sidecar（暂时存起来，后面再放）

VLM只负责补attributes、geometry hints、forward relations。**别让VLM从零生成整个graph**，让它填空——它填空比生成稳定100倍。最终边集：

$$E = E_{\text{parent}} \cup E_{\text{vlm}} \cup E_{\text{wall}} \cup E_{\text{corner}} \cup \text{Inv}(\cdot)$$

$\text{Inv}(\cdot)$ 是加逆向边（A是B的parent → 加一条B是A的child的边）。这样graph变成双向可查的。

**Intuition**: 确定的部分（骨架）用规则推，模糊的部分（语义、属性）让VLM推。这分工特别对。

### 3.2 Layout Code（Stage 3-4）：核心innovation在这

**Stage 3** 是整个paper最漂亮的环节。流程是：

```
生成初始layout code → render成top-down图 → VLM critique → sanitize → 修代码 → 循环
```

公式化：

$$
\begin{aligned}
C^{(0)} &= \text{Generate}(I, D_1, G) \\
R^{(t)} &= \text{Render}(C^{(t-1)}) \\
(A^{(t)}, s_t) &= \text{Critique}(I, R^{(t)}, G) \\
\widetilde{A}^{(t)} &= \text{Sanitize}(A^{(t)}, D_1, G) \\
C^{(t)} &= \text{Revise}(C^{(t-1)}, \widetilde{A}^{(t)})
\end{aligned}
$$

- $C^{(t)}$：第$t$轮的layout code
- $R^{(t)}$：把上一轮代码render出来的top-down图
- $A^{(t)}$：VLM critic的textual feedback（"少了个沙发"、"床撞墙了"、"桌椅关系错了"）
- $s_t$：critic打的分（综合object coverage、overlap、boundary、relation）
- $\widetilde{A}^{(t)}$：**sanitize后的feedback**——这是关键

**Sanitize是干什么？** critic有时候会说"把这面墙挪一下"或者"加个门"。但墙和门在Stage 1就定死了。如果照着改，下一轮critic一看input image对不上又让改回去——infinite loop。Sanitize就是把critic的feedback跟 $D_1$ 和 $G$ 对一下，**把不支持的建议过滤掉**，只保留合理的小修小改。

终止条件：$s_t \geq s^\star$（分数够了）或 $t = T_{\text{max}} = 5$（最多5轮）。

**Stage 4** 在major layout冻住之后，append wall-mounted物体和visible的小物件。Tiny surface-bound小物件（书、杯子）defer到后面Stage 6+，因为那时候桌面表面已经生成出来了，可以精准贴上去。

### 3.3 Object Code（Stage 5-6）：把方块替换成真家具

**Stage 5**：拿 $C_{\text{layout}}$ parse出每个物体的position/size/orientation，用这些layout attributes去**ground** VLM。VLM对每个物体输出fine-grained description $D_{\text{FU}}$（color, material, function, structure, style），外加一个global room style JSON。

**为什么要layout-grounded？** 你直接问VLM"图里那个table长啥样"，它会泛泛而谈"棕色木桌"。你告诉它"在(x,y,θ)有一个1.2m×0.6m的table"，它会结合周围环境给出更precise的描述，比如跟旁边chair的style匹配。

**Stage 6**：把每个object拆成semantic parts：

$$\mathcal{P}_i = \Phi_{\text{geo}}(o_i, d_i) = \{p_{i,j}\}_{j=1}^{K_i}$$

- $d_i \in D_{\text{FU}}$：物体$i$的描述
- $p_{i,j}$：第$j$个part，包含primitive type（box/cylinder/sphere）、semantic name（seat/backrest/leg）、local size、offset、rotation
- $K_i$：物体$i$的part数量

**关键design**：parts定义在**原proxy的local frame**里，所以生成的object自动继承coarse layout的pose——layout consistency不会在geometry refinement阶段被破坏。

然后做code replacement：

$$C_{\text{geom}} = \text{Replace}(C_{\text{layout}}, \{o_i \mapsto \mathcal{P}_i\}_{i=1}^N)$$

**复杂小物件**用retrieval策略：先放placeholder占位，然后从asset library $\mathcal{B}$ 里检索最匹配的asset替换：

$$b^\star = \arg\max_{b \in \mathcal{B}} \text{match}(b; \text{label, description, placeholder\_size})$$

match score同时考虑semantic relevance和size compatibility。

### 3.4 Decoration（Stage 8-10）：材质、贴图、灯光

这阶段做一件事：**只rewrite或append code，绝对不动placement和geometry**。把appearance和geometry彻底解耦——避免加material的时候意外把layout搞坏。

- **Stage 8**: part-level PBR materials，预测material type、linear-RGB base color、roughness、metallic、specular。Glass/mirror用shader override。
- **Stage 9**: 大面积或带纹样的表面（floor、wall、rug、painting）用image generation model合成texture map，inject到material node graph里。
- **Stage 10**: VLM推断lighting style（主光源方向、窗户自然光、人工光源、ambient intensity），转成Blender light objects。

然后是**deterministic post-hoc correction**，最核心的是collision/boundary fix：

$$\mathbf{x}_i^\star = \arg\min_{\mathbf{x} \in \mathcal{N}(\hat{\mathbf{x}}_i)} \|\mathbf{x} - \hat{\mathbf{x}}_i\|_2 \quad \text{s.t.} \quad B(o_i, \mathbf{x}) \subseteq B_{\text{room}}, \quad B(o_i, \mathbf{x}) \cap B(o_j) = \varnothing$$

- $\hat{\mathbf{x}}_i$：VLM给的position（可能撞墙或撞别的物体）
- $\mathcal{N}(\hat{\mathbf{x}}_i)$：原位置附近的local grid neighborhood
- $B(o_i, \mathbf{x})$：物体$i$在位置$\mathbf{x}$的bounding box
- $B_{\text{room}}$：房间边界
- collision constraint只对**nearby non-parent objects**生效——parent-child关系允许overlap（桌上放杯子天经地义）

实现就是deterministic local search + boundary clamping + stacking offset。这个projection保证最后的scene在物理上至少是合法的（没穿墙、没穿插）。

## 4. Cross-Stage Memory：治"健忘症"

长pipeline最容易死在哪？**后面stage忘了前面stage干嘛的**。

CaR维护一个shared memory $\mathcal{M}$，每个stage $s$ 产出一个typed artifact：

$$\mathcal{M}_s = \mathcal{M}_{s-1} \oplus e_s, \quad e_s = \langle s, \tau_s, O_s, \eta_s \rangle$$

- $e_s$：stage $s$的artifact
- $\tau_s$：artifact类型（"scene_graph"、"layout_code"、"object_profile"等）
- $O_s$：output内容
- $\eta_s$：metadata

**关键design**：每个downstream stage**只读一个predefined memory view**，不读整个memory。两个好处：
1. 减少prompt noise（Stage 5不需要看Stage 1的perimeter scan细节）
2. 减少hallucinated dependencies（避免stage之间脑补出虚假因果）

Ablation验证：去掉memory，Layout IoU从73.2%暴跌到58.0%。**forgetting是真实的killer**。

## 5. 实验告诉我们什么

### 5.1 Table 1 的核心信号

最striking的对比：
- **Gemini 3.1-Pro单pass**: Sim 2.0, Use 0.0, Accept 1.0——基本废物
- **CaR w/ Gemini 3.1-Pro**: Sim 9.0, Use 8.0, Accept 7.5——expert acceptable

这个jump说明：**harness的价值远大于base model的raw capability**。弱模型+好harness完爆强模型+naive prompting。

**为什么Gemini在CaR下比GPT-5.5好？** 看spatial reasoning指标：
- Gemini3.1-pro w/CaR: Rotation Acc 93.6%, Support Acc 94.0%, Spatial Relation 79.8%
- GPT-5.5 w/CaR: Rotation Acc 92.2%, Support Acc 80.1%, Spatial Relation 71.4%

Gemini在structured spatial reasoning上明显更强（top-down view的rotation、support关系理解），GPT-5.5更偏holistic generation。**Spatial reasoning是top-down-to-3D任务的核心能力**，所以Gemini更适合。

### 5.2 对比VIGA

VIGA的Light score 8.0跟CaR持平，但Sim只有5.5，Use只有4.5。VIGA在lighting上做了专门优化，但layout preservation和practical usability弱——模板化场景，细节缺失，物体位置不准。

CaR的structured harness恰恰解决的就是layout preservation和细节保留。

### 5.3 Visual Feedback的sweet spot

| Feedback iter | Obj. Recall | Layout IoU | Rotation Acc |
|---|---|---|---|
| 0 | 33.8% | 64.0% | 71.9% |
| 3 | 35.6% | 65.7% | 73.2% |
| 5 | 38.4% | 66.2% | 75.4% |
| 10 | 39.1% | 64.2% | 72.6% |

0→5单调上升，10反而下降。**典型over-correction现象**：每次critic的feedback都有noise，迭代太多agent就开始chasing noise而不是真正improve layout。5次是个甜点。

## 6. 我的几个intuition

**Intuition 1: Agentic system的可靠性来自constraint + decomposition，不是来自base model brute force**

VLM一次性生成整个3D scene code这事儿太ill-posed，模型再聪明也搞不定。CaR把任务拆成10个well-posed的小问题，每个都在VLM sweet spot里，再用memory串起来，再用sanitize把不合理的feedback过滤掉——这是**约束求解**的思想，跟纯generation是两码事。

**Intuition 2: Code作为scene representation是个好选择**

Code executable（可以直接render拿visual feedback）、editable（设计师可以直接改code）、compositional（parts可以复用）、interpretable（debug能看懂）。这跟neural field、diffusion-based 3D generation这种black-box输出有本质区别。对interior design这种**需要iterative refinement**的应用，editability是硬需求。

**Intuition 3: Top-down view是spatial prior的最佳载体**

Floor plan为啥在建筑设计里用了几百年？因为它**正交地编码了所有spatial信息**——没perspective distortion，没occlusion，物体数量、相对位置、墙的拓扑一目了然。CaR用top-down image作为input，直接bypass了text-to-3D的spatial ambiguity。

**Intuition 4: Sanitize feedback比feedback本身更重要**

VLM critic会幻觉，会suggest不合理的修改。如果不sanitize，agent会陷入infinite loop。这其实是个general principle：**agentic system的critic必须constrained by task的hard constraints**，否则critic自己就是instability来源。

**Intuition 5: Future direction——video model作为neural renderer**

CaR生成的3D scene是procedural geometry，photorealism有限。但它的3D scene可以作为strong structural prior，喂给video generation model做appearance refinement。这相当于**两阶段neural rendering**：deterministic code保证structure，neural model保证appearance。Limitations里也提到这个方向。video model现在>5秒trajectory保持不了temporal consistency，但如果有3D scene作为prior，这个问题可能能缓解。

## 7. 给你的一段超浓缩总结

CaR是个10-stage agentic pipeline，把top-down image → Blender code这个ill-posed问题拆成scene graph构造 → layout code render-compare循环 → object geometry replacement → material/lighting refinement四个well-posed的子问题，中间用cross-stage memory防forgetting，用sanitized critic feedback防infinite loop，用post-hoc projection防物理违反。实验证明：**harness的design远比base model的raw capability重要**，Gemini 3.1-Pro从"基本废物"到"expert acceptable"全靠这个harness。

## 8. 参考链接

- **Code-as-Room project page**: https://code-as-room.github.io/
- **VIGA (baseline)**: https://arxiv.org/abs/2601.11109
- **Holodeck**: https://arxiv.org/abs/2312.09067
- **ProcTHOR**: https://arxiv.org/abs/2206.06994
- **SceneWeaver**: https://arxiv.org/abs/2411.19091 (NeurIPS 2026 文中标注)
- **SAGE**: https://arxiv.org/abs/2602.10116
- **SceneSmith**: https://arxiv.org/abs/2602.09153
- **3D-GPT**: https://arxiv.org/abs/2310.12974
- **LayoutGPT**: https://arxiv.org/abs/2305.15393
- **Gemini 3.1 Pro Model Card**: https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf
- **GPT-5.5 System Card**: https://deploymentsafety.openai.com/gpt-5-5/gpt-5.5.pdf

---

# Code-as-Room 深度技术解析

## 1. 问题动机与核心 Intuition

这篇paper要解决的问题非常清晰: 给一张**top-down view**的房间图片 $I$, 生成可执行的 **Blender code** $C$, 使得这段code在Blender里跑起来能重建出一个完整、可编辑、可渲染的3D室内场景。

$$C = \mathcal{A}(I)$$

这里 $\mathcal{A}$ 是整个agentic harness, $I$ 是top-down image, $C$ 是Blender可执行代码。

**为什么是top-down而不是perspective?** Top-down view天然编码了**global spatial prior**: 物体数量、相对位置、房间结构、walls/doors/windows的拓扑。Perspective image虽然视觉丰富但有occlusion和perspective distortion, 对于**holistic room generation**反而不利。这其实是仿照人类设计师的工作流 —— 设计师都是从floor plan开始的。

**为什么是code而不是JSON/scene graph?** Code是**executable + editable + compositional**的。JSON只能描述layout, 但无法表达procedural geometry (比如一个沙发的part decomposition)、material node graph、lighting setup。Code作为representation的好处是: 你可以直接render它来获取visual feedback, 这就闭环了。这跟VIGA [32] 的思想一致, 但CaR解决了VIGA在top-down场景下**infinite loop**和**spatial detail loss**的问题。

## 2. 核心架构: 10-Stage Pipeline

整个pipeline的intuition是**coarse-to-fine decomposition**, 把一个ill-posed的image-to-3D问题拆成多个well-posed的子问题, 每个子问题对VLM来说都是tractable的。

### 2.1 Coarse Stage (Stage 1-4): 解决"放什么、放哪"

**Stage 1: Spatial Semantic Analysis**

$$D_1 = F_1(I, P_1), \quad \mathcal{M} = \mathcal{M} \oplus D_1$$

- $D_1$: schema-constrained description, 包含functional zones, object hierarchies, architectural elements
- $P_1$: perimeter-aware prompt (扫描walls, corners, openings, coarse grid来recover peripheral和wall-mounted objects)
- $\mathcal{M}$: cross-stage memory
- $\oplus$: memory append操作

这里有个关键的design choice: 每个object被赋予identifier, category, placement type (floor/wall/surface-mounted), parent (如果有hierarchy关系)。Walls/doors/windows/openings被保留为**fixed spatial references** —— 这意味着后面stage不能随意改动它们, 只能围绕它们布局。

**Stage 2: Object-centric Scene Graph Construction**

这一步从 $D_1$ 推出一个**deterministic skeleton**:

$$S = \{V_{\text{arch}}, V_{\text{major}}, E_{\text{parent}}, M_{\text{minor}}\}$$

- $V_{\text{arch}}$: architectural features (walls, doors, windows)
- $V_{\text{major}}$: layout-defining objects (主要家具)
- $E_{\text{parent}}$: hierarchy-derived relations (parent-child, 比如桌面上的物体)
- $M_{\text{minor}}$: minor objects sidecar (小物件, 暂时存起来后面处理)

然后VLM只负责**complete attributes, geometry hints, forward relations**, 而不是从头构建图 —— 这把VLM的生成空间压缩了, 减少hallucination。最终的edge set:

$$E = E_{\text{parent}} \cup E_{\text{vlm}} \cup E_{\text{wall}} \cup E_{\text{corner}} \cup \text{Inv}(\cdot)$$

- $E_{\text{vlm}}$: VLM补充的语义关系
- $E_{\text{wall}}$, $E_{\text{corner}}$: wall-anchor关系 (物体靠墙、靠角)
- $\text{Inv}(\cdot)$: 逆向关系 (A是B的parent → B是A的child)

**Intuition**: 这里把"确定性的部分"(skeleton)和"需要VLM推理的部分"(attributes, relations)分开, 是一个非常聪明的design。VLM擅长semantic reasoning但不擅长structured generation, 所以你给它一个骨架让它填空, 比让它从零生成整个graph稳定得多。

### 2.2 Layout Code Generation (Stage 3-4): Render-and-Compare 闭环

**Stage 3: Major Layout with Visual Feedback**

这是整个paper最核心的innovation之一。核心是一个**render-critique-revise loop**:

$$
\begin{aligned}
C^{(0)} &= \text{Generate}(I, D_1, G) \\
R^{(t)} &= \text{Render}(C^{(t-1)}) \\
(A^{(t)}, s_t) &= \text{Critique}(I, R^{(t)}, G) \\
\widetilde{A}^{(t)} &= \text{Sanitize}(A^{(t)}, D_1, G) \\
C^{(t)} &= \text{Revise}(C^{(t-1)}, \widetilde{A}^{(t)})
\end{aligned}
$$

变量解释:
- $C^{(t)}$: 第 $t$ 次迭代的layout code
- $R^{(t)}$: 把 $C^{(t-1)}$ 渲染成top-down image
- $A^{(t)}$: VLM critic输出的textual feedback (missing objects, overlaps, boundary violations, relation errors)
- $s_t$: VLM评估的layout quality score (object coverage, overlap, boundary consistency, spatial relation correctness的综合)
- $\widetilde{A}^{(t)}$: sanitized feedback —— 关键步骤, 用 $D_1$ 和 $G$ 过滤掉critic可能提出的"unsupported architectural changes"
- $T_{\text{max}} = 5$: 最大迭代次数

**为什么Sanitize很重要?** VLM critic有时候会suggest "把这面墙挪一下"或者"加一个门" —— 但这些architectural elements在Stage 1就fixed了。如果不sanitize, agent会陷入**修改架构→违反input image→重新修改**的infinite loop。这恰恰是VIGA在top-down场景下的failure mode。

终止条件: $s_t \geq s^\star$ (quality score达标) 或 $t = T_{\text{max}}$。

**Stage 4: Auxiliary Layout**

在major layout frozen的基础上, append wall-mounted objects和visually salient minor objects:

$$M_{\text{minor}}^\star = \{m \in M_{\text{minor}} \mid m \text{ is visible and not surface-bound}\}$$

这里有个很重要的filtering: 只保留**visible at coarse layout scale**且**not surface-bound**的minor objects (rugs, floor lamps, plants, large decorations)。Tiny surface-bound objects (books, cups, small tabletop items)被defer到Stage 5+的fine-grained placement。

$$C_{\text{layout}} = \text{Append}(C_{\text{layout}}^{\text{major}}, M^\star)$$

### 2.3 Fine Stage (Stage 5-6): 解决"长什么样、几何细节"

**Stage 5: Layout-grounded Object Description**

把 $C_{\text{layout}}$ parse成placed objects, 用它们的layout attributes (position, size, orientation)来ground VLM。VLM输出:

$$D_{\text{FU}} = U_{\text{FU}}(I, C_{\text{layout}}, \mathcal{M})$$

$D_{\text{FU}}$ 包含每个object的: color, material, function, structure, style。同时输出一个**global room-style description JSON** $s_{\text{room}}$。

**Intuition**: 这一步的关键在于**layout-grounded**。如果你直接让VLM描述图像里的物体, 它会描述得很general ("a brown wooden table")。但如果你告诉它"在位置(x, y, θ)有一个1.2m×0.6m的table", VLM就能给出更precise、更context-aware的描述, 比如考虑到周围物体的style consistency。

**Stage 6: Object Geometry Replacement**

对每个placed object $o_i$, geometry agent预测一个**semantic 3D geometry primitive decomposition**:

$$\mathcal{P}_i = \Phi_{\text{geo}}(o_i, d_i) = \{p_{i,j}\}_{j=1}^{K_i}$$

变量解释:
- $d_i \in D_{\text{FU}}$: object $i$ 的fine-grained description
- $p_{i,j}$: 第 $j$ 个part, 包含 primitive type (box, cylinder, sphere, etc.), semantic part name (seat, backrest, leg), local size, offset, rotation
- $K_i$: object $i$ 的part数量

**关键design**: parts定义在**original proxy的local frame**里, 所以生成的object自动继承coarse-layout的pose。这保证了layout consistency不会在geometry refinement阶段被破坏。

然后做code replacement:

$$C_{\text{geom}} = \text{Replace}(C_{\text{layout}}, \{o_i \mapsto \mathcal{P}_i\}_{i=1}^N)$$

**Tiny objects的hybrid策略**: 对于visually distinctive但procedurally难以生成的物体, 用retrieval:

$$b^\star = \arg\max_{b \in \mathcal{B}} \text{match}(b; \text{label, description, placeholder\_size})$$

- $\mathcal{B}$: asset library
- match score联合考虑semantic relevance和size compatibility
- 选中的asset被scaled和aligned到placeholder位置, 保留support surface和footprint

### 2.4 Interior Decoration (Stage 8-10): 解决"看起来怎么样"

**Geometry-preserving code rewriting chain**:

$$C_{\text{obj}} \xrightarrow{\text{ApplyMat}} C_{\text{mat}} \xrightarrow{\text{ApplyTex}} C_{\text{tex}} \xrightarrow{\text{RenderSetup}} C_{\text{raw}}$$

每一步只rewrite或append Blender code, **不修改object placement或geometry**。这是一个很重要的约束 —— 把appearance和geometry解耦, 避免在加material的时候意外破坏layout。

**Stage 8: Material Assignment** —— part-level PBR materials, 预测material type, linear-RGB base color, roughness, metallic value, specular strength。Glass和mirror用shader overrides。

**Stage 9: Texture and Decorative Surfaces** —— 用high-capacity image generation model合成texture maps (floors, walls, rugs, paintings, posters), 注入到material node graph。Planar decorative elements有explicit UV mapping。

**Stage 10: Lighting, Rendering, Post-hoc Correction**

VLM推断overall lighting style (dominant illumination direction, window-driven natural light, artificial light sources, ambient intensity), 转换成Blender light objects和renderer settings。

然后是**deterministic post-hoc correction pass**, 其中最重要的就是collision/boundary fix:

$$\mathbf{x}_i^\star = \arg\min_{\mathbf{x} \in \mathcal{N}(\hat{\mathbf{x}}_i)} \|\mathbf{x} - \hat{\mathbf{x}}_i\|_2 \quad \text{s.t.} \quad B(o_i, \mathbf{x}) \subseteq B_{\text{room}}, \quad B(o_i, \mathbf{x}) \cap B(o_j) = \varnothing$$

变量解释:
- $\hat{\mathbf{x}}_i$: VLM生成的position (可能有boundary或overlap violation)
- $\mathcal{N}(\hat{\mathbf{x}}_i)$: $\hat{\mathbf{x}}_i$周围的local grid neighborhood
- $B(o_i, \mathbf{x})$: object $i$ 在位置 $\mathbf{x}$ 的bounding box
- $B_{\text{room}}$: room boundary
- collision constraint只对**nearby non-parent objects**生效 (parent-child关系的物体允许overlap, 比如桌上放杯子)

这个projection实际上通过**deterministic local search + boundary clamping + stacking offsets for supported objects**实现。最终:

$$C = \text{PostHoc}(C_{\text{raw}})$$

## 3. Cross-Stage Memory: 解决Context Forgetting

Agent-based framework最头疼的问题就是**long context → forgetting**。CaR的解法是维护一个shared memory:

$$\mathcal{M}_s = \mathcal{M}_{s-1} \oplus e_s, \quad e_s = \langle s, \tau_s, O_s, \eta_s \rangle$$

- $e_s$: stage $s$ 产出的typed artifact
- $s$: stage identifier
- $\tau_s$: artifact type (e.g., "scene_graph", "layout_code", "object_profile")
- $O_s$: output content
- $\eta_s$: metadata

**关键design**: 每个downstream stage只读一个**predefined memory view**, 而不是整个memory。这有两个好处:
1. 减少prompt noise (不让Stage 5看到Stage 1的所有细节)
2. 减少hallucinated dependencies (避免stage之间产生虚假的因果链)

Table 3的ablation验证了这一点: 去掉memory后, Layout IoU从73.2%暴跌到58.0% —— 因为later stages无法可靠地reuse early stage的image-derived信息, 导致missing objects和layout drift。

## 4. Benchmark设计: 4个维度评估

这是一个很有价值的contribution, 因为之前没有专门针对**code-based 3D room synthesis**的benchmark。

**Benchmark suite**: 41个scenes, 覆盖:
- Room types: bedrooms, kitchens, living rooms (Simple/Middle/Hard by spatial scale & object density)
- Specialized scenes: laboratories, barber shops, cafes
- Image styles: photorealistic photos, synthetic renderings, abstract line drawings

**4个evaluation dimensions**:

| Dimension | Metrics | Intuition |
|-----------|---------|-----------|
| Visual Understanding | Obj. Recall, Func. Acc. | 能不能识别出图里有什么 |
| Spatial Reasoning | Self Overlap, Layout IoU, Spatial Relation, Rotation Acc., Support Acc. | 位置/朝向/支撑关系对不对 |
| Code Generation | Agent Completion Rate, Exec. Rate | Pipeline能不能跑完, code能不能在Blender里执行 |
| Scene Quality | Image Similarity, Scene Usability, Aesthetic Quality | 最终效果好不好 |

**Annotation pipeline**: 因为ground truth unavailable for diverse inputs, 用human-in-the-loop: Gemini 3.1先生成coarse labels → human annotators用reverse code refinement tool同步visual edits和scene code。

## 5. 实验结果深度分析

### 5.1 Table 1: Benchmark主结果

几个关键观察:

**Direct generation vs. CaR**:
- GPT-5.5 direct: Exec Rate 42.2%, Agent Completion 71.1% → CaR: Exec Rate 73.3%, Agent Completion 71.1% (completion不变但exec提升)
- Gemini3.1-pro direct: 几乎unusable → CaR: Exec Rate 95.5%, Agent Completion 100%

**为什么Gemini在CaR下表现比GPT-5.5好?** 从Table 1看:
- Gemini3.1-pro w/CaR: Rotation Acc 93.6%, Support Acc 94.0%, Spatial Relation 79.8%
- GPT-5.5 w/CaR: Rotation Acc 92.2%, Support Acc 80.1%, Spatial Relation 71.4%

GPT-5.5在visual understanding (Obj. Recall 67.5% vs 55.5%)和aesthetic (7.52 vs 8.20)上略弱, 但在spatial reasoning上差距明显。这可能是因为Gemini在**structured spatial reasoning**上更强 (比如理解top-down view的rotation和support关系), 而GPT-5.5更偏向holistic generation。

**Self Overlap**: CaR w/ Gemini3-flash 只有2.57%, 而GPT-5.5 direct是14.5%。这主要归功于Stage 10的post-hoc correction (那个projection公式)。

### 5.2 Table 2: Human Evaluation

20个experts, 4个维度: Similarity, Usability, Lighting, Acceptability。

最striking的对比:
- Gemini3.1-Pro / Single-pass: Sim 2.0, Use 0.0, Accept 1.0 (基本不可用)
- CaR w/ Gemini3.1-Pro: Sim 9.0, Use 8.0, Accept 7.5

这个jump说明了**agentic harness的价值远大于base model capability**。一个弱模型 + 好的harness >> 强模型 + naive prompting。

**VIGA对比**: VIGA的Light score是8.0 (跟CaR持平), 但Sim只有5.5, Use只有4.5。这说明VIGA在lighting上做了专门优化, 但layout preservation和practical usability弱 —— 这正是CaR的structured harness要解决的问题。

### 5.3 Table 3: Ablation Studies

**Memory mechanism**:
- w/o Memory: Obj. Recall 48.2%, Layout IoU 58.0%, Rotation Acc 88.4%
- Full Model: Obj. Recall 55.5%, Layout IoU 73.2%, Rotation Acc 93.6%

Layout IoU掉了15.2% —— 这说明memory对**maintaining spatial consistency across stages**至关重要。没有memory, later stages会"忘记"earlier stage的layout决策, 导致drift。

**Visual feedback iterations**:
- 0 iter: Obj. Recall 33.8%, Layout IoU 64.0%, Rotation Acc 71.9%
- 3 iter: 35.6%, 65.7%, 73.2%
- 5 iter (Ours): 38.4%, 66.2%, 75.4%
- 10 iter: 39.1%, 64.2%, 72.6%

0→5 iter单调提升, 但10 iter反而下降。这是典型的**over-correction / layout drift**现象。Intuition: 每次critic的feedback都有noise, 迭代太多会让agent追逐noise而不是真正improve layout。

## 6. 与VIGA的深度对比

VIGA [32] 是这篇paper最直接的baseline。VIGA的核心是**analysis-by-synthesis loop** for perspective images, 但当naively扩展到top-down view时有几个failure modes:

1. **Fine-grained spatial detail loss**: VIGA的pipeline没有显式的coarse-to-fine decomposition, 直接生成整个scene code, 导致top-down view里的小物体容易被忽略
2. **Infinite loop**: VIGA的critic没有sanitize步骤, 可能suggest architectural changes, 导致agent反复修改无法收敛
3. **Context forgetting**: VIGA没有cross-stage memory, 长workflow里early stage的决策会被forget

CaR的解法对应这三个问题:
1. Coarse-to-fine: 先fix layout, 再enrich geometry/appearance
2. Sanitize critic feedback w.r.t. $D_1$ and $G$
3. Cross-stage memory with typed artifacts and predefined views

## 7. Scene Re-rendering: 一个有趣的Future Direction

Section 4.4展示了用GPT-5.5对Blender-rendered scenes做image-level re-rendering。这其实揭示了一个**两阶段neural rendering**的思想:

1. **Structural prior**: CaR生成的3D scene提供room structure, object layout, spatial relations, camera-consistent geometry
2. **Appearance refinement**: 用image/video generation model增强materials, lighting, object details

这个pipeline的好处是: structural consistency由deterministic code保证, 而appearance的photorealism由neural model保证。两者解耦, 各自发挥优势。

Limitations里提到的**video generation models作为neural renderers**是一个很有前景的方向 —— 目前的video model在>5秒的trajectory上还难以保持temporal consistency, 但如果有了3D scene作为strong prior, 这个问题可能被缓解。

## 8. 技术联想与延伸

### 8.1 与LayoutGPT [6], Holodeck [30]的对比

这些都是text-driven的方法。LayoutGPT用LLM生成CSS-like layout, Holodeck用LLM生成scene graph然后retrieve assets。它们的核心limitation是: **text description无法precisely specify spatial information**。你说"a bedroom with a bed near the window", LLM不知道"near"是10cm还是1m, 也不知道window在哪面墙。CaR用top-down image作为input, 直接bypass了这个问题。

### 8.2 与3D-GPT [22], ShapeAssembly [10]的对比

3D-GPT用LLM做procedural 3D modeling (Blender), ShapeAssembly学习生成3D shape structure的program。这些都是**object-level**或**local structure**的code generation, 而CaR是**room-scale**的。Room-scale的挑战在于: 全局layout consistency, multi-object coordination, architectural constraints。

### 8.3 与SceneWeaver [27], SAGE [25], SceneSmith [19]的对比

这些都是agentic scene generation, 但都是text/task-driven。SceneWeaver用self-reflective agent, SAGE用generator-critic, SceneSmith用hierarchical VLM agents。CaR的区别在于: **image-conditioned** + **code representation** + **structured harness**。

### 8.4 与Embodied AI的联系

ProcTHOR [5] 用procedural generation大规模生成interactive houses for embodied AI training。CaR生成的3D rooms是**editable and executable**的, 理论上可以作为embodied AI的training environments。特别是CaR支持diverse room types (labs, barber shops, cafes), 这对embodied AI的generalization很有价值。

### 8.5 Code-as-Scene paradigm的更深层意义

CaR和VIGA代表的code-as-scene paradigm其实是一种**neural-symbolic**方法: 
- Symbolic部分: Blender code, executable, editable, interpretable
- Neural部分: VLM做perception, reasoning, code synthesis

这种paradigm的好处是: 生成的scene是**fully editable**的 (你可以直接改code来调整), 而neural field或diffusion-based 3D generation的输出往往是black-box。对于interior design这种需要iterative refinement的应用, editability是critical的。

## 9. 个人思考与Critical Analysis

**Strengths**:
1. **Structured harness design**: coarse-to-fine + memory + sanitized feedback, 每一个design choice都有明确的problem correspondence
2. **Comprehensive benchmark**: 4个维度, 41个scenes, human-in-the-loop annotation
3. **Ablation studies**: 验证了memory和visual feedback各自的贡献, 以及feedback iterations的sweet spot

**Potential Weaknesses**:
1. **Top-down view limitation**: 只能处理top-down input, 对arbitrary view不支持。真实世界的reference image往往是perspective的
2. **Procedural geometry fidelity**: 很复杂物体 (比如雕花椅子、复杂灯具)用primitive decomposition难以忠实表达, 不得不依赖asset retrieval
3. **Evaluation的subjectivity**: Scene Quality metrics依赖VLM-based scoring和human study, 可能存在bias
4. **Scalability**: 10-stage pipeline的latency和cost可能较高, paper没report inference time

**Open questions**:
- 能否把top-down view扩展到multi-view input?
- 能否用更少stages达到similar quality? (比如把Stage 1-2合并, Stage 8-9合并)
- Memory mechanism能否用RAG-style retrieval代替predefined views, 以支持更flexible的cross-stage reasoning?
- 能否把post-hoc correction的projection步骤integrate到Stage 3的render-critique-revise loop里, 形成统一的constraint satisfaction framework?

## 10. 参考链接

- **Project page**: https://code-as-room.github.io/
- **VIGA (baseline)**: https://arxiv.org/abs/2601.11109
- **Holodeck**: https://arxiv.org/abs/2312.09067 (CVPR 2024)
- **ProcTHOR**: https://arxiv.org/abs/2206.06994 (NeurIPS 2022)
- **SceneWeaver**: NeurIPS 2026
- **SAGE**: https://arxiv.org/abs/2602.10116
- **SceneSmith**: https://arxiv.org/abs/2602.09153
- **3D-GPT**: https://arxiv.org/abs/2310.12974 (3DV 2025)
- **ShapeAssembly**: https://arxiv.org/abs/2011.10247 (SIGGRAPH Asia 2020)
- **LayoutGPT**: https://arxiv.org/abs/2305.15393 (NeurIPS 2023)
- **Gemini 3.1 Pro**: https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf
- **GPT-5.5**: https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf
- **LLplace**: https://arxiv.org/abs/2406.03866

---

**核心takeaway**: 这篇paper的intuition是: **complex generation task的可靠性来自于structured decomposition + explicit memory + constrained feedback**, 而不是来自于base model的raw capability。一个well-designed agentic harness可以让Gemini3.1-Pro从"几乎不可用"变成"达到expert human-acceptable quality"。这背后的思想 —— 把ill-posed problem分解成多个well-posed sub-problems, 每个sub-problem都在VLM的capability sweet spot内 —— 是agentic system design的一个general principle, 远超3D room generation这一个应用。
