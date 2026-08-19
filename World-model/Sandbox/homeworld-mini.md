---
source_pdf: homeworld-mini.pdf
paper_sha256: 018cd8b2c926231978a4942ba6f2d27b152af9d234cdf9df9f4357d774059b72
processed_at: '2026-08-19T11:24:25-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HomeWorld 人话版

Andrej，简单说，这篇 paper 就是**教 AI 装修一套完整的房子**——从画户型图到摆家具到放杯子，全流程一条龙。

---

## 一句话概括

**用 LLM 画户型图 → 用图像生成模型摆大件家具 → 用 VLM 反复检查纠错 → 最后往桌面塞小物件**，四步走完一整套房。

---

## 为什么这事难

想象你让 AI 直接一次性生成一个完整的家。问题立刻冒出来：

- **3D 数据太少**：全世界公开的真实 3D 室内场景，ScanNet 才 90 套，Matterport3D 也就 90 套，3D-FRONT 6.8K 套还是单间 curated 的。这点数据根本 train 不出一个能生成千变万户完整家的 3D diffusion model。
- **2D 户型图数据多但乱**：RPLAN 80K、LI-FULL 5M 张图，但都是 2D 的，不是 sim-ready 的。
- **Rule-based 方法死板**：ProcTHOR 靠 hand-crafted 规则生成，validity 没问题但千篇一律，看十个家像同一个家。
- **2D-lifting 方法几何漂移**：Text2Room 这种从 2D text-to-image 生成 3D 的，多视角拼起来几何就乱了，像拼图拼不上。

HomeWorld 的核心 insight：**别想着一步到位，把"装房子"拆成人类设计师的四个阶段，每个阶段用最合适的 prior**。

---

## 四步走，人话版

### Step 1：LLM 画户型图

**人话**：告诉 LLM "我要三室一厅，主卧带卫生间，厨房连餐厅"，让它吐一个 JSON 文件描述户型。

**关键 trick**：不直接让它输出房间多边形坐标（因为 LLM 算坐标会算歪、房间会重叠），而是用一个 **K-D tree** 把空间递归切分——先竖切一刀，再横切一刀，每次切下来的小块再切，最后每个叶子节点贴一个房间类型标签。

**为什么这招管用**：K-D tree 天然保证房间不重叠（因为是切蛋糕式的 partition，每一刀都把空间严格二分），把"房间不能重叠"这个约束从 loss function 搬到了 representation 里。就像你用 quaternion 表示 rotation 就天然避开 gimbal lock 一样——representation 本身就消除了一类错误。

训练数据：自己爬了 1.08M 张真实 floorplan 图，用 detector + OCR 提取 vectorized 结构，最终 314K 干净样本，fine-tune Qwen3-4B。

效果：拓扑有效性 87.7%（vs ProcTHOR 72.6%），拓扑多样性 22.4 种（vs 6.2），user study 第一。

### Step 2：图像模型摆大件

**人话**：户型图有了，先把 bed、sofa、dining table 这些大件摆好。

**怎么做**：
1. 先在 Blender 里把户型图 instantiate 成一个**空壳**（walls + floor + ceiling + 基础灯光）
2. 从**俯视角**渲染空壳，用图像 inpainting 模型往里"画"家具——bed 画这里、sofa 画那里
3. 用 SAM-3 提取 instance mask，用 SAM-3D 把 2D 物体 lift 成 3D bounding box
4. 再切到**人眼视角**（ego-centric），在房间边界走一圈，补小物件（挂钩、橱柜上层、垃圾桶这些俯视角看不到的）

**为什么空壳是 hidden gem**：Text2Room 失败是因为没有 3D anchor，多视角生成时几何飘了。HomeWorld 先把 floorplan 实例化成空壳，相当于给图像生成模型画了一个"画框"——你只能在框内画，画出来自然贴在墙上、落在地上，不会飘。

**视角选择是 greedy set cover**：把地板划 grid，每个 candidate 视角覆盖一部分 grid cell，贪心选"当前覆盖最没看过的区域最多"的视角，直到整个房间被覆盖。这是经典 set cover 近似，保证 $1 - 1/e$ 近似比。

### Step 3：VLM 反复纠错

**人话**：前两步生成的东西肯定有问题——沙发挡住门了、椅子穿墙了、桌子飘在半空。让一个 VLM 当"质检员"，看一眼发现问题，提一个修改建议（把沙发往左挪 0.5 米），执行，再看，再改，循环到没问题为止。

**为什么需要这步**：2D inpainting 没有 3D 几何 prior，透视会失真；SAM-3D 在 occlusion 下几何会 incomplete；多视角 inpainting 之间会不一致。这些 error 是 open-loop generation 的固有缺陷，必须 closed-loop 修。

**训练数据怎么造**：
1. 拿干净的 layout，故意注入错误（沙发撞墙、门被堵、椅子穿墙）
2. Oracle 算法对每个错误状态算"最佳修复动作"（translate / rotate / combined）
3. 训完初版 refiner 后，让它自己 rollout 在 corrupted layout 上，收集它自己产生的中间状态，oracle 重新 label，加回训练集——这就是 **DAgger** 的思想，缩小 train 和 deploy 的 distribution gap

**公式直觉**：每步 action 的 score：
$$
\text{score}(a) = -w_1 \cdot \text{残差误差} - w_2 \cdot \text{新引入的违规} - w_3 \cdot \text{移动成本} + w_4 \cdot \text{语义合理性}
$$

选 score 最高的动作执行。这个设计的妙处在于：refiner 不是一次性判断"对不对"，而是像一个 plumber 一步步拧螺丝，每步只做最小改动，避免 over-correction。

### Step 4：桌面塞小物件

**人话**：大件摆好了，但桌面是空的——书桌上没书没笔，餐桌上没碗没筷，橱柜里没盘子。这步专门往支撑面上放小物件，让场景"有生活气息"。

**为什么单独成阶段**：
- **Scale gap**：大件是米级，小物件是厘米级，同一分辨率 canvas 上画不清
- **Coordinate gap**：大件在 room frame，小物件在 furniture frame（书在桌上，桌在房间里），需要显式 support relation
- **Use case gap**：embodied AI 任务（抓取、开门、摆桌子）几乎全在小物件层面

**物理属性**：用 PhysX-Anything 预测每个小物件的 density $\rho$、Young's modulus $E$、Poisson's ratio $\nu$，算 mass $m = \rho \cdot V_{\text{mesh}}$，让物件在 simulator 里能真实 simulate 抓取、推拉、掉落。

---

## 为什么这套设计 work

### 核心思想：分层 + 每层换 prior

| 阶段 | 任务性质 | 最合适的 prior |
|---|---|---|
| Floorplan | Topological + symbolic（房间数、邻接关系） | LLM（token 序列 + K-D tree） |
| Furniture layout | Spatial + visual（沙发朝哪、茶几离多远） | Image foundation model（billions of indoor photos） |
| Refinement | Error correction | VLM closed-loop（verifier-refiner agent） |
| Small objects | Surface affordance + physics | Inpainting + PhysX-Anything |

这对应到认知科学里的 **symbolic vs subsymbolic 划分**——floorplan 是符号推理，家具摆放是视觉常识，两者用不同的 prior 最自然。

### 为什么不 end-to-end

End-to-end 学一个 3D diffusion 直接从 text 生成 whole-home scene 理论上最优雅，但：
1. 数据不够（3D scene 太少）
2. 计算太贵（whole-home 是 10m × 10m × 3m 的 volume）
3. 控制粒度不够（没法精确控制"主卧带卫生间"这种 topological constraint）

HomeWorld 的 modular approach 是**pragmatic 的妥协**——在 3D data 真正 scale up 之前，用 2D foundation prior + explicit 3D constraint + VLM closed-loop 是更 sample-efficient 的 path。

---

## 和现实装修的类比

| 装修阶段 | HomeWorld 对应模块 |
|---|---|
| 画施工图 | LLM + K-D tree 生成 floorplan |
| 水电入场前空房子 | Unfurnished shell in Blender |
| 摆大件家具（床、沙发、餐桌） | Top-down view roaming |
| 填小件（挂钩、橱柜） | Ego-centric roaming |
| 设计师反复检查"这沙发挡门了" | VLM recursive refiner |
| 软装摆件（书、花瓶、餐具） | Manipulable object placement |
| 给家具贴材质、装灯 | Shell texture + lighting setup |

整个 pipeline 就是把人类设计师的工作流显式化、自动化。

---

## 数据集的贡献

这个被低估了。300K 真实 floorplan + 5K furnished whole-home scene，公开 release，是社区大福利：

- **Floorplan 维度**：RPLAN 80K 是当前主流，HomeWorld 300K 几乎 4x
- **Sim-ready**：3D-FRONT 6.8K 是 sim-ready 但单间级，HomeWorld 5K 是 whole-home + manipulable objects >15 per scene
- **Hybrid curation**：real floorplan + generative obj，比纯 rule-based（ProcTHOR）更 diverse，比纯 reconstruction（ScanNet）更 clean

---

## 可能的局限

读 paper 时我注意到几个潜在问题：

1. **Inpainting 用 Gemini 3.0 Image Pro Preview**（"Nano Banana Pro"），不是开源的，社区复现得换 FLUX 或 SD3 inpainting，效果可能掉。
2. **SAM-3D 几何质量**没报告 reconstruction error，embodied simulator 里 mesh 质量直接影响 physics engine stability。
3. **Refiner iteration limit** 没明说，hard case（多 collision 堆叠）可能 fix 不完。
4. **5K furnished scene** 对训下游 RL policy 可能还是偏少（虽然 paper 强调是 generation pipeline 不是 dataset paper）。
5. **Style bias**：image inpainting 受 base model prior 限制，可能偏向 western interior style，亚洲风格、其他文化背景的风格可能 underrepresent。

---

## 最终 Intuition

HomeWorld 的 thesis：**全屋场景生成的 bottleneck 是 data，解法不是硬端到端，而是把任务拆成符号层和视觉层，每层用最合适的 prior，再用 closed-loop 把 error 修掉**。

这和 AlphaProof（Lean + LLM）、Voyager（Minecraft + LLM agent）、Reflexion 这些工作的哲学一脉相承：**在 data 不够的时候，用 strong prior + explicit constraint + closed-loop correction 比 end-to-end 学习更 sample-efficient**。

对你做 embodied AI 的 intuition 来说，这套 pipeline 模式可能比具体的 K-D tree trick 或 inpainting 细节更有借鉴价值——**modular design + closed-loop refinement** 是 data-scarce regime 下的通用 pattern。

Project page: https://kairos-homeworld.github.io

希望这人话版帮你快速抓住 essence，Andrej。如果你想 drill down 到任何一个 module 的具体实现细节（比如 K-D tree 的 token 化 schema、VLM refiner 的 reward 设计、SAM-3D grounding 的几何 alignment 数学），我可以再展开。

---

# HomeWorld: 从 Floorplan 到可仿真整屋场景的层次化生成

Andrej, 这篇 paper 本质上做的事情是把"室内场景生成"这个长期被局部方法割裂的任务，重构成一个 **floorplan → unfurnished shell → furniture → manipulable objects** 的四阶段 sequential pipeline，每一阶段都用强 prior（LLM / foundation image model / VLM refiner）来 anchor 输出，再用 explicit 3D geometry constraint 把 2D 自由生成 tether 回物理空间。下面我尽量把 intuition 和技术细节都铺开讲。

Project page: https://kairos-homeworld.github.io

---

## 1. Big picture: 为什么这是一个 layered cake

室内场景生成的核心矛盾，简单概括为：

- **3D residential data 稀缺**：ScanNet ~90 个 home，Matterport3D ~90 个 home，3D-FRONT ~6.8K 但都是 curated 的 single room-level scene，reconstruction-based data 又有 incomplete geometry + fragmented surface 的问题。
- **2D floorplan data 量大但 representation 混乱**：LI-FULL 有 5M、RPLAN 80K、MSD 17K，但这些都不是 sim-ready 的。
- **Rule-based（ProcTHOR / Infinigen / Structured3D）保证 validity 但缺乏 diversity**，asset library 受限。
- **End-to-end 3D generative（DiffIndScene / Blockfusion）**计算贵、难 scale 到 whole-home。
- **2D-lifting（Text2Room / HouseCrafter / LucidDreamer）**自由度太高，drifting geometry，几何 inconsistent。

HomeWorld 的取舍是 **分层 + 每层换 prior**：
- Layer 1（floorplan）：用 LLM autoregressive，因为 floorplan 本质是 topological + 语义结构，用 token 序列表示最自然。
- Layer 2（furniture layout）：用 foundation image model 的 2D prior，因为家具 co-occurrence、风格 commonsense 在 2D image corpus 里远比 3D scene corpus 丰富。
- Layer 3（refinement）：用 VLM 做 closed-loop correction，因为 synthesis + grounding 必然 error-prone，需要一个 verifier-refiner agent 把布局 push 到 feasible manifold 上。
- Layer 4（small manipulable objects）：用 surface-conditioned inpainting + 物理 attribute 推理，让场景具备 embodied interaction 的 affordance。

这种分层对应到 cognitive 的 "coarse-to-fine placement"：人类设计师先画 partition → 再摆大件 → 再放小件 → 再调对齐，pipeline 把这个流程显式化。

---

## 2. Floorplan Dataset Curation（300K 数据怎么来）

这是一个被低估的工程贡献。流程五步（paper §3.1）：

1. **Image collection**：从在线 real-estate portfolio 抓 1.084M 张 floorplan 图像（含 doors / windows / walls / dimension annotations）。
2. **Detection-based vectorization**：train 一个 detector 在 2K 手标图像上，识别 doors / windows / walls / dimension chains。bounding box → centerline，附近 endpoint merge → reconstruct vectorized structure。
3. **OCR**：抽 room label + 尺寸标注。
4. **Quality filtering**：丢掉 noisy sample，最终 314K validated floorplan。
5. **Topological + caption annotation**：从 shared doors + adjacent walls 推 connectivity，给所有 entity（room/door/window/wall）唯一 ID，自动生成 detailed textual caption。

**关键 insight**：caption 的随机组合（room counts / boundary info / positional constraints / connectivity / attachment requirements）构成 LLM 训练 prompt-target pairs，而且同一个 floorplan 可以 sample 出 multiple difficulty levels 的 prompt。这相当于 self-data-augmentation 通过 prompt perturbation 实现。

参考数据集：
- RPLAN: https://github.com/ywl0715/RPLAN-Dataset
- LI-FULL: https://www.nii.ac.jp/dsc/idr/en/liful1/1.html
- MSD: https://github.com/SZuces/MSD
- ResPlan: https://arxiv.org/abs/2508.14006

---

## 3. K-D Tree Representation：把 floorplan 序列化的核心 trick

这是 paper 在 representation 上的关键设计选择，§3.1 末段和 §4.3 ablation 验证了它的价值。

### 3.1 为什么用 K-D tree 而不是 polygon

直接让 LLM 输出 polygon 坐标 $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$ 的问题：

- polygon overlap：两个房间多边形会重叠，需要后处理 clipping，error-prone。
- holes：apartment outline 内部出现 unaccounted 空洞。
- LLM 对连续坐标数值精度敏感，小数点错位 → 房间分离/disconnected。

K-D tree 把 interior space 递归 partition：每次沿 axis-aligned 切一刀（垂直或水平），内部节点存 `(split_axis, split_coordinate)`，叶节点存 `(room_id, room_type)`。附件（doors / windows）单独存。

形式化：一棵 K-D tree $T$ 递归定义为
$$
T = \begin{cases}
\text{Leaf}(r_i, t_i) & \text{leaf node: room } r_i \text{ of type } t_i \\
\text{Internal}(a, c, T_L, T_R) & a \in \{V, H\}, c \in \mathbb{R}, T_L, T_R \text{ subtrees}
\end{cases}
$$

- $a$：split axis，$V$ = vertical cut, $H$ = horizontal cut
- $c$：split coordinate（normalized 到 apartment bounding box 内 $[0,1]$）
- $T_L, T_R$：左右子树（在 $a$ 方向上 $<c$ 和 $\geq c$ 的两个 subregion）

序列化为 JSON 之后是 hierarchical text，**天然 compatible with LLM autoregressive decoding**。

### 3.2 为什么避免 geometric error

K-D tree 的 spatial partition **结构性保证 non-overlapping**：任意两个 leaf 的 bounding region 互斥且 union 覆盖 parent。这把 "polygon overlap" 的几何约束从 loss function 里搬到了 representation 里——这是一种 **implicit constraint injection via representation design**，类似于把 SO(3) rotation 用 quaternion 表示 vs Euler angle：representation 本身消除了一类错误。

### 3.3 Ablation（§4.3）

| Representation | Overlap% | Holes% | Disconnected% |
|---|---|---|---|
| Canonical (polygon) | 9.2 | 2.0 | 16.2 |
| **K-D tree (Ours)** | 0 (by construction) | 0 (by construction) | 12.3 |

Disconnected 仍 12.3% 是因为开门/连接信息是 attachment level 的，不直接由 K-D tree 保证，但比 polygon 下降 3.9 个百分点。

---

## 4. LLM-based Floorplan Generator

### 4.1 Training 配置

- Base model: **Qwen3-4B-Instruct**（参考 https://github.com/QwenLM/Qwen3）
- SFT + LoRA（参考 https://arxiv.org/abs/2106.09685）
- 数据 split：281K train / 15K val / 15K test

### 4.2 Task formulation

把 floorplan synthesis 当 **constrained instruction-following**。Prompt 包括：

- a) apartment dimensions（外轮廓尺寸）
- b) room count & types（如 2 bedroom + 1 living room + 1 kitchen + 1 bathroom）
- c) door/window placement 要求
- d) 强制 K-D tree axis-aligned 输出结构
- e) 房间 adjacency & connectivity 约束

### 4.3 评估指标

**Graph Validity**：生成 floorplan 的 connectivity graph 是 fully connected 的比例。所谓 connectivity graph $G = (V, E)$：
- $V$：room 集合
- $E$：两 room 之间通过 door 相连的边

fully connected 意味着 agent 从任意 room 都能 navigable 到任意其他 room（不出现 isolated room）。

**Graph Diversity**：固定 room 数 $n$ 时，生成样本中 distinct connectivity graph 的数量。分 typed（含 room type 标签）和 untyped（仅 topology）。

### 4.4 Quantitative Results（Table 2）

| Method | Graph Validity | Typed Div. | Untyped Div. |
|---|---|---|---|
| ProcTHOR-10K | 72.6% | 6.2 | 2.3 |
| **Ours (Qwen3-4B)** | **87.7%** | **22.4** | **6.9** |

Graph validity +15.1%，diversity 大约 3-4x 提升。这说明：在真实 floorplan 数据上 SFT 的 LLM 学到的远比 ProcTHOR 的 procedural rule 更 rich、更接近真实拓扑。

### 4.5 User Study（Table 3）

vs FloorPlan-LLaMA & Floorplan-Diffusion：

| Method | Reasonability | Richness | Avg |
|---|---|---|---|
| FloorPlan-LLaMA | 1.85 | 2.00 | 1.93 |
| Floorplan-Diffusion | 2.13 | 1.81 | 1.97 |
| **Ours** | **2.20** | **2.34** | **2.27** |

Reasonability 和 Richness 都第一，Richness 优势尤其大（2.34 vs 1.81/2.00），印证 K-D tree + LLM 在拓扑多样性上的收益。

---

## 5. Hierarchical Furnishing: Top-Down + Ego-Centric Roaming

这一节（§3.2）是 paper 的"trick 集合"，把 2D-lifting 的几何不一致问题用 explicit 3D shell constraint 强行压住。

### 5.1 Unfurnished Shell as 3D Anchor

先生成 empty shell（在 Blender 里 instance），含 wall/floor/ceiling + 基础 lighting/material。作用：

- 提供 **3D constraint**：相机视角下任何生成 object 必须 ground 到 shell geometry 内，不能 drift。
- 提供 **lighting prior**：rendered shell 已经带 shading，后续 inpainting 自然有 consistent 阴影方向。

对比 Text2Room（https://github.com/lukashoel/text2room）的 free-roaming：他们没有 explicit 3D anchor，导致 geometry drifting（多视角生成的 mesh 在不同 view 下位置漂移）。

### 5.2 Top-Down View Roaming

核心 idea：大件 furniture（bed/sofa/dining table）的 placement 主要由 **global constraints** 决定（room boundary / door / window / inter-room connectivity），所以先用 top-down view 抓 global layout。

流程：
1. Render empty shell top-down → $I_{\text{top}}$
2. Annotate $I_{\text{top}}$ with door + window 位置（door 还标注 adjacent room，因为门通向哪个房间决定 furniture 朝向）
3. Image inpainting model（Gemini 3.0 Image Pro Preview / "Nano Banana Pro"）合成 furnished top view $\hat{I}_{\text{top}}$
4. Lightweight validation 丢弃 infeasible 生成
5. VLM 做 open-vocab category recognition + **SAM-3**（https://arxiv.org/abs/2511.16719）提取 instance mask + 2D box
6. 2D box → 3D bounding box（语义 label + 物理尺寸 + pose）

### 5.3 Ego-Centric View Roaming

Top-down 的局限：occlusion + scale ambiguity → 小物件、wall-attached 物件（bins / hooks / racks / upper-lower cabinets）无法恢复。引入 ego-centric roaming。

**Viewpoint selection (heatmap-based greedy)**：

把 floor plane 离散化为 grid，candidate camera poses 在 room boundary + interior 沿 fixed FoV + safe distance to walls 采样。每个 candidate view 覆盖 grid cell 的子集，累加得 coverage heatmap $H \in \mathbb{R}^{G \times 1}$，其中 $G$ = grid cell 数。

Greedy 选择第 $k$ 个 viewpoint：
$$
v_k^* = \arg\max_{v \in \mathcal{V}_{\text{cand}}} \sum_{g} \mathbb{1}[\text{uncovered}] \cdot \text{coverage}_{v}(g)
$$

等价于：选择当前**最少观测区域贡献最大**的视角，更新 heatmap。这是经典 set cover 的 greedy 近似，coverage submodular 保证 $1 - 1/e$ 近似比。

**Ego-centric synthesis + 3D grounding**：

对每个 viewpoint $k$：
1. Render ego-centric image $I_k$
2. Inpainting → $\hat{I}_k$（add secondary objects）
3. SAM-3 instance mask
4. **SAM-3D**（https://arxiv.org/abs/2511.16624）reconstruct 3D geometry
5. Geometric alignment module 把新 3D asset integrate 进 room

多视角拼接 → 360° coverage → densely populated 3D environment。

---

## 6. Recursive Layout Refinement（VLM-based closed-loop）

§3.3 是 paper 设计上最巧妙的部分。问题：2D synthesis 和 3D grounding 都 error-prone（blocked doorways / boundary-crossing / collisions / floating objects）。需要 closed-loop verifier。

### 6.1 Sequential Decision-Making Formulation

把 refinement 建模为 MDP：
- **State** $s_t$：当前 3D layout（top-view render + structured 3D layout description，含 category/position/size/rotation/room-level architecture）
- **Action** $a_t$：structured corrective action，$(\text{target\_obj\_id}, \text{transform})$，transform 可以是 translation $T$、rotation $R$、或 combined
- **Transition** $s_{t+1} = f(s_t, a_t)$：deterministic apply 到 3D bounding box layout
- **Termination**：通过 validation 或达到 iteration limit

VLM 在每一步：
1. 接收 $(s_t, \text{observation})$ 多模态输入
2. Eval under：physical plausibility / semantic consistency / structural constraints
3. Predict $a_t$
4. Apply → re-render → re-check
5. Loop

这本质上是 **LLM/VLM as an agent for layout repair**，和最近 Reflexion / Self-Refine / Voyager 这些 self-correction 工作 同构。

### 6.2 Fine-tuning 数据构造

三个 component 互补：

**(a) Corrupted layout construction**：从 clean layout 注入 controlled perturbation（boundary violation / collision / doorway blocking / implausible rotation / scale inconsistency / category-level error）。记录每一步 perturbation → 自然形成 reverse trajectory。

**(b) Oracle-labeled repair actions**：对每个 corrupted state，oracle 提出候选 corrections：translation, rotation, combined transform, displacement along minimum separation direction。每个 candidate 用 verifier 评分：
$$
\text{score}(a) = w_1 \cdot (-\text{residual\_err}) + w_2 \cdot (-\text{new\_violations}) + w_3 \cdot (-\text{movement\_cost}) + w_4 \cdot \text{semantic\_plausibility}
$$

选 score 最高的作为 GT next action。

**(c) Model-in-the-loop samples**：训完初版 refiner 后，rollout 在 corrupted layouts 上，收集 model 自己产生的中间 state → oracle relabel → 加回训练集。这相当于 DAgger（Dataset Aggregation, https://arxiv.org/abs/1011.0686）的思想，缩小 supervised training 和 deployment 的 distribution gap。

每个 training sample：`(top-view rendering, 3D layout description, current error state, optional action history, oracle-labeled correction action)`。

### 6.3 Ablation 验证

去掉 VLM refiner：CR 从 0.05 → 0.20，OOB 从 0.01 → 0.05，证明 refiner 是 physical plausibility 的关键 guard。

---

## 7. Manipulable Object Placement（§3.4）

大件 furniture 落位后，桌面 / 柜面 / 床头仍然空，既不真实又缺乏 embodied interaction affordance。

### 7.1 Surface Object Synthesis

1. 识别可 support 小物件的 furniture：依据 semantic category + geometric property（horizontal surface + sufficient area）
2. 对每个 selected furniture，构造 local surface canvas，conditioned on (room_type, furniture_category, target_support_region)
3. Inpainting model synthesize surface-level arrangements（desk → books/laptop/lamp/stationery；kitchen countertop → bowls/bottles/utensils）

### 7.2 Physical Attribute Assignment（PhysX-Anything）

用 **PhysX-Anything**（https://arxiv.org/abs/2511.13648）预测每个 object：
- semantic category
- dimensions
- part decomposition
- material properties: density $\rho$, Young's modulus $E$, Poisson's ratio $\nu$

Mass estimation：
$$
m = \rho \cdot V_{\text{mesh}}
$$

其中 $V_{\text{mesh}}$ 是 mesh volume，$\rho$ 是预测的密度。

Part-level material 用于 component-wise simulation params（rigid body / articulation / friction）。

### 7.3 3D Layout Recovery & Filtering

VLM 把 surface image parse 成 manipulable objects（category / scale / relative position / orientation），先在 furniture local coordinate 表示，再通过 furniture 的 pose + dimension transform 到 global room coordinate：
$$
\mathbf{p}_{\text{global}} = R_{\text{furniture}} \cdot \mathbf{p}_{\text{local}} + \mathbf{t}_{\text{furniture}}
$$

其中 $R_{\text{furniture}}$ 是 furniture rotation matrix，$\mathbf{t}_{\text{furniture}}$ 是 furniture translation，$\mathbf{p}_{\text{local}}$ 是 object 在 furniture 局部坐标系的位置。

**Physical filtering**：移除/调整：
- 超出 support boundary
- 严重 inter-object collision
- 浮空（floating above support surface）
- 违反 category-level support constraint

**保留 support relation**：每个 small object 与 host furniture 的 explicit 关系，方便下游 scene graph 构建 / task generation / embodied interaction。

---

## 8. Experiments：Furniture Layout Generation（§4.2）

### 8.1 Baselines

- LayoutGPT (https://arxiv.org/abs/2305.15393) - LLM-based，限 living room + bedroom
- Holodeck (https://github.com/allenai/Holodeck) - LLM-based，whole-home
- LayoutVLM - VLM-based

### 8.2 Metrics

- **CR (Collision Ratio)**：严重 inter-object penetration（除 semantically valid contact 外）比例。↓
- **OOB (Out-of-Boundary)**：object footprint 超出 room boundary 比例。↓
- **VD (Volume Density)**：$\frac{\sum_i \text{Vol}_i}{\text{Floor Area}}$，反映空间填充度。↑
- **FOD (Footprint Object Density)**：$\frac{\text{Object Count}}{\text{Union Area of Furniture Footprints}}$，反映小物件密度。↑

### 8.3 Results (Table 4, 4 room types × 25 samples each)

| Method | CR↓ | OOB↓ | VD↑ | FOD↑ |
|---|---|---|---|---|
| LayoutGPT | 0.05 | 0.01 | 0.22 | 1.99 |
| Holodeck | 0.07 | 0.02 | 0.22 | 2.10 |
| LayoutVLM | 0.20 | 0.01 | 0.27 | 2.15 |
| **Ours** | **0.05** | **0.01** | **0.35** | **4.16** |

CR/OOB 持平最好，VD +59%，**FOD 几乎 2x**，证明 manipulable object placement 阶段把场景从 sparse-elegant 推到 cluttered-realistic。

### 8.4 Ablation (Table 6)

| Variant | CR↓ | OOB↓ | VD↑ | FOD↑ |
|---|---|---|---|---|
| w/o Top-down View Roaming | 0.09 | 0.02 | 0.22 | 2.6 |
| w/o Ego-centric View Roaming | 0.07 | 0.02 | 0.20 | 3.44 |
| w/o VLM Refiner | 0.20 | 0.05 | 0.33 | 3.02 |
| w/o Manipulable Object Placement | 0.05 | 0.01 | 0.33 | 1.82 |
| **Ours (full)** | 0.05 | 0.01 | 0.35 | 4.16 |

Intuition：
- **Top-down** 提供 global structure prior，去掉后 VD 从 0.35→0.22（与 baseline 持平），说明 top-down 是把 2D foundation prior 转化成 3D layout 起点的关键桥梁
- **Ego-centric** 主要做 furniture-level enrichment，去掉后 VD 0.35→0.20
- **VLM Refiner** 是 physical plausibility 的硬 guard，去掉后 CR 4x 恶化
- **Manipulable Object** 主要拉 FOD（4.16→1.82），VD 小幅提升（0.35→0.33）

### 8.5 User Study (Table 5, 30 users)

Reasonability / Aesthetics / Complexity 三维：

| Method | Reasonability | Aesthetics | Complexity | Avg |
|---|---|---|---|---|
| LayoutGPT | 0.202 | 0.184 | 0.153 | 0.180 |
| Holodeck | 0.395 | 0.412 | 0.452 | 0.420 |
| LayoutVLM | 0.252 | 0.260 | 0.283 | 0.265 |
| **Ours** | **0.807** | **0.827** | **0.797** | **0.811** |

Ours 在所有维度接近 0.8，第二 Holodeck 0.42，差距巨大。注意 Complexity 上 Ours 0.797 vs LayoutGPT 0.153，说明 LayoutGPT 输出太 sparse，HomeWorld 的 density + manipulable objects 让 perceived complexity 显著提升。

---

## 9. 关于数据集 vs 现有 (Table 1)

关键对比项：

| Dataset | Source | Scope | Sim-ready | Floorplan | Furnished | MObj |
|---|---|---|---|---|---|---|
| RPLAN | Real | Home | × | 80K | - | - |
| ScanNet | Real(Rec) | Home | × | - | 90H | n/r |
| 3D-FRONT | Designed | Home | ✓ | 6.8K | 6.8K H | n/r |
| Structured3D | Designed | Home | × | 3.5K | 21K S | n/r |
| InternScenes | Collection | Mixed | Partial | - | 48K S | 8 |
| SceneVerse | Collection | Mixed | Partial | - | 68K S | n/r |
| ProcTHOR-10K | Synthetic(Rules) | Home | ✓ | 10K | 10K H | n/r* |
| **Ours** | FP(Real)+Obj(Gen) | Home | ✓ | **300K** | **5K H** | **>15** |

Ours 的特征：**hybrid curation**（real floorplan + generative obj），明确承诺 sim-ready + manipulable objects >15 per scene，对齐 embodied AI 的训练需求。

参考链接：
- 3D-FRONT: https://3dfront.org
- Structured3D: https://structured3d-dataset.github.io/
- ProcTHOR: https://procthor.allenai.org/
- InternScenes: https://github.com/InternScenes/InternScenes
- SceneVerse: https://sceneverse.github.io/
- ScanNet: http://scannet.cs.princeton.edu/
- Matterport3D: https://niessner.github.io/Matterport/

---

## 10. 一些 Intuition 上的关键点

### 10.1 为什么 floorplan 阶段用 LLM 而 furniture 阶段用 image model

Floorplan 是 **topological + symbolic** 结构：room count、adjacency、door/window placement 这些是离散决策，且和自然语言高度 aligned（"three bedrooms with one bathroom adjacent to the master bedroom"）。LLM autoregressive + token 序列输出 + K-D tree 序列化是天然匹配。

Furniture layout 是 **spatial + visual** 结构：bed 放哪个角、sofa 朝向哪、coffee table 离 sofa 多远，这些是 visual-commonsense 问题，image foundation model（trained on billions of indoor photos）的 prior 比 LLM 更 strong。

这种"任务性质决定 prior 类型"的分层对应到 cognitive 科学里的 symbolic vs subsymbolic 划分。

### 10.2 为什么需要 closed-loop refinement

Open-loop generation（不管多强的 prior）必然有 errors，因为：
1. 2D inpainting 没有 3D 几何 prior，透视失真 → 3D box grounding 错位
2. SAM-3D reconstruction 在 occlusion 下几何 incomplete
3. 多视角 inpainting 之间不一致

把 VLM 当 verifier-refiner agent，每步观察 + 决策 + 执行 + 重新观察，本质上把 generation 的 manifold projection 拆成多步 feasible direction 移动。这个思路和机器人 planning 里的 RRT-star + obstacle projection、和 LLM self-correction（Reflexion, Self-Refine, Voyager, https://arxiv.org/abs/2303.17651, https://arxiv.org/abs/2303.17666）同构。

### 10.3 为什么 surface-centric manipulable object placement 单独成阶段

- **Scale gap**：大件 furniture（米级）和小件 manipulable objects（厘米级）在同一个 inpainting canvas 上分辨率打架
- **Coordinate system gap**：大件在 room frame，小件在 furniture frame，需要 explicit support relation
- **Use case gap**：embodied AI 任务（pick-and-place, open drawer, set table）几乎全在小物件层面，需要 explicit affordance 和 physics

这就是为什么 PhysX-Anything 单独预测 $\rho, E, \nu$ 三个 material params，让小物件在 simulator 里能 simulate grasp、push、drop 的真实物理响应。

### 10.4 Unfurnished shell 是 hidden gem

Paper 在 §3.2 之前快速带过 shell 构造，但其实是把 2D-lifting 做对的关键。Text2Room 失败的 root cause 是没有 3D anchor，视角之间几何 drift。HomeWorld 把 shell（floorplan 的 3D 实例化）作为 explicit constraint，让 inpainting 模型只能在 shell 内画东西，自然把几何 tether 在 floorplan 给的空间里。这类似于 depth-conditioned generation（ControlNet depth）的思路，把 explicit geometric constraint 从 loss 推到 input。

### 10.5 Dataset 的重要性被严重低估

300K 真实 floorplan + caption 是一个非常稀缺的资源。FloorPlan-LLaMA 用 collection data，RPLAN 80K 都不算大。300K 让 LLM SFT 有足够样本学 real-world topological distribution。Paper 承诺 release 这个 dataset，对 community 是大贡献。

---

## 11. 一些 Limitations 和 Potential Concerns

Paper 没有显式 limitation section，但读出来的潜在问题：

1. **Inpainting model 黑箱依赖**：用 Gemini 3.0 Image Pro Preview（"Nano Banana Pro"），不是开源 model，复现成本高。未来社区可能要换成 FLUX/SD3 inpainting 重新跑。
2. **SAM-3D 几何质量**：小物件 3D reconstruction 在 paper 里没报告 reconstruction error，embodied simulator 里 mesh 质量直接影响 physics engine stability。
3. **Refiner iteration limit**：paper 没明确说 iteration cap，对 hard case（多 collision）可能 fix 不完。
4. **5K furnished scene vs 300K floorplan**：furnished scene 数量仍小，对训练下游 RL policy 可能不够（虽然 paper 强调是 generation pipeline）。
5. **Style diversity**：image inpainting 受 base model prior 限制，可能 bias 向 western interior style，亚洲风格、低 income 地区风格可能 underrepresent。
6. **Manipulable object 的 asset replacement**：§3.4 说用 3D generative model 替换 asset，但具体哪个模型、replacement 后是否保持 surface alignment，paper 描述粗略。

---

## 12. 与相关工作的图谱

| 方向 | 代表工作 | 链接 |
|---|---|---|
| Floorplan LLM | FloorPlan-LLaMA | https://arxiv.org/abs/2503.xxxxx |
| Floorplan diffusion | Floorplan-Diffusion | - |
| Rule-based 3D home | ProcTHOR / Infinigen | https://procthor.allenai.org, https://infinigen.org |
| Diffusion scene | DiffIndScene, PhyScene, ChOrD | https://arxiv.org/abs/2403.07414, https://physcene.github.io/ |
| 2D lifting | Text2Room, HouseCrafter, LucidDreamer | https://github.com/lukashoel/text2room |
| LLM scene agent | Holodeck, LayoutGPT, Worldcraft, I-Design | https://github.com/allenai/Holodeck, https://arxiv.org/abs/2502.15601 |
| SAM family | SAM-3, SAM-3D | https://arxiv.org/abs/2511.16719, https://arxiv.org/abs/2511.16624 |
| Physics asset | PhysX-Anything | https://arxiv.org/abs/2511.13648 |
| LLM base | Qwen3 | https://github.com/QwenLM/Qwen3 |
| PEFT | LoRA | https://arxiv.org/abs/2106.09685 |
| DAgger (model-in-the-loop) | DAgger | https://arxiv.org/abs/1011.0686 |

---

## 13. 总结性 Intuition

HomeWorld 的核心 thesis：**全屋场景生成的 bottleneck 是 data，而 data 缺的不是规模而是合适的 representation 和 prior 的分层组合**。它做的几件事：

1. 把 floorplan 用 K-D tree token 化，让 LLM 在 symbolic 空间生成
2. 把 furniture layout 用 image prior + 3D shell anchor，让 foundation model 在 visual 空间生成同时被 3D 几何约束
3. 把 refinement 用 VLM closed-loop，把 error correction 建模成 sequential decision
4. 把 manipulable object placement 单独阶段化，配 physics-aware grounding

这套 layered + closed-loop 的 pipeline 工程上很重，但每一步都解决了 specific failure mode（topology error / drift / collision / empty surface），是 modular approach 而非 end-to-end 的胜利。这给整个 embodied AI scene generation 领域指了一个方向：在 3D data 真正 scale up 之前，**2D foundation prior + explicit 3D constraint + VLM closed-loop** 是 pragmatic 的 path，比直接端到端学 3D diffusion 更 sample-efficient。

社区可期待的是 300K floorplan + 5K furnished scene 的 release，以及 pipeline 各 module 被 community 替换/升级后的 variants。

Project page: https://kairos-homeworld.github.io

希望对你 build intuition 有帮助，Andrej。如果你想 drill down 到任何具体 module（比如 K-D tree decoding 的 token 化细节、VLM refiner 的 reward 设计、SAM-3D grounding 的几何 alignment 数学），我可以再展开。
