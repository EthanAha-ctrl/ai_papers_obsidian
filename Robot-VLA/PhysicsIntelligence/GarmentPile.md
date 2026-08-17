---
source_pdf: GarmentPile.pdf
paper_sha256: 975d9bad96cce39b5fcfd27d5d45b448ad7144283b5b7bb13012b25fed2f58e6
processed_at: '2026-08-04T12:08:54-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 GarmentPile

Andrej，前面讲得太学术了，我用更直觉的方式重新过一遍核心 idea。

---

## 这个 paper 在解决什么生活问题

你家里洗衣机的衣服堆在一起，你想让 robot 一件一件拿出来放到 basket 里。听起来简单，实际上坑特别多：

- 你抓一件 shirt 的 corner，整件 shirt 的另一头掉地上了 → unclean
- 你抓上面那件，结果它袖子穿进下面那件的领口，把下面的也拖出来了 → entangle
- 衣服堆得太乱，根本找不到一个"能成功抓"的点 → 需要先扒拉几下再抓

所以这个 paper 的核心就一句话：**在 tangled 的衣服堆里，每个 3D point 都打一个分，表示"抓这个点能不能成功"；如果全场都是低分，就先 affordance-guided 地 pick-place 一下 reorganize 场景，直到出现高分点再抓**。

---

## 为什么 previous approach 会塌

previous approach 有两条 symbol-driven 的路线：

### 路线 A：先分割每件衣服，再推理关系

典型例子 Support-M，用 SAM 把每件衣物 segmentation 出来，然后找"不被其他衣物支撑"的那件抓。

问题：garment 在 clutter 里严重 occlusion + 严重 deformable，SAM 根本分不对。paper Appendix Figure 6 给了 case：SAM 把"半件 shirt"当成"一整件"输出了。这其实暴露了一个 ill-posed 问题——garment clutter 里"什么算一件衣服"本身就模糊，半截袖子露在外面算独立一件吗？

### 路线 B：用 VLM 直接选点

典型例子 GPT-Fabric-M，把 RGB + depth 丢给 GPT-4o，prompt 它输出 optimal pixel 坐标。

结果：success rate 0.4 左右，跟 random 差不多。paper Appendix C 把 prompt 都贴出来了，GPT-4o 基本就是"随机指一个点"。这告诉我们 VLM 在 rigid clutter 上还行，一到 deformable + heavy occlusion 的场景，geometric precision 就崩溃了。

---

## 他们的 core idea：per-point affordance，回避 segmentation

回避"分割"这个 ill-posed 问题。直接 input raw point cloud（N×3，不带 RGB），output 一个 N 维的 score map，每个 score 表示"在这个 point 上执行 retrieval 能不能成功"。

success 的定义是复合的：既不落地，也不拖累邻居。这个 label 由 simulator 自动给——直接执行动作看物理结果，不需要 human annotation。

关键 insight：per-point feature 自然会从 local 到 global aggregation。PointNet++ 的 set abstraction 层一层一层 widen 邻域：

- 第 1 层看的是 local curvature → 这个点 gripper 夹不夹得住
- 第 2 层看的是中距离结构 → 这件衣服整体什么形状
- 第 3 层看的是全局 context → 这件和其他衣物什么关系

paper 不显式设计这三层 module，让 supervision signal 自己 shape 出来。可视化 Figure 4 / 8 显示 model 真的学到了：

- 衣服 middle 区域 high score，corner 低分 → 因为抓 corner 容易让对侧落地
- 上面那件衣服 high score，下面那件低分 → 因为抓下面会拖上面
- 两件衣物交界处低分 → 因为 gripper 会同时夹住两件

这跟 NeRF 的精神一样：结构由 loss 塑造。

---

## 最 tricky 的设计：cascade supervision

adaptation（reorganize）需要 pick 一个点 + place 到另一个点。如果直接学 joint affordance $A(p_{pick}, p_{place})$，action space 是 $N \times N$，学不动。

他们把它 disentangle 成 cascade，而且训练顺序很关键：**retrieval → place → pick**，每一级用前一级的 model 当 oracle 提供 label。

### 为什么这个顺序

直觉是：**离物理结果越近的，越先学**。

- **retrieval model** 离物理结果最近：直接执行 retrieval action，simulator 立刻告诉你成功还是失败，label 干净。
- **place model** 离物理结果 1 step 远：给定 pick 点，place 到某个位置，scene 变成新状态 $O'$，用已经 trained 的 retrieval model 评估 $O'$ 的 affordance 是否 improve。这就是 1-step lookahead value function。
- **pick model** 离物理结果 2 step 远：给定 pick 点，用 trained place model 找最优 $p_{place}$，再执行，用 retrieval model 评估。2-step lookahead。

这其实是把 RL 的 value iteration 用 supervised BCE 替代了——simulator 当 oracle，每一级 distill 下一级。每级 model 就是 depth-1 / depth-2 / depth-0 的 Q function。

类比一下：
- AlphaZero 的 self-play 用 MCTS 当 oracle 提供 policy label
- DAgger 用 expert 当 oracle
- 这里用 retrieval model 当 oracle 给 place / pick 提供 label

---

## Adaptation 怎么 trigger

empirical 判据：

$$P_{high} = \frac{\#\{p : A_p > 0.9\}}{N}$$

$P_{high} < 0.1$ 就 trigger adaptation。0.1 是经验值，paper 没做 sensitivity analysis，但 Table 5 显示 1-2 轮 adaptation 后 success rate 就 saturate，所以这个 threshold 大致 work。

关键 ablation：3 轮 random adaptation success rate 只有 0.719，跟 0 轮（0.712）几乎一样。说明 random stirring 没用，adaptation 的智能全在 pick / place affordance 上。

这个 ablation 我觉得很 elegantly 说明了一个 point：reorganize 本身不够，必须有 purposeful 的 reorganization，否则你扒拉半天场景只是变得更乱。

---

## Sim-to-real 关键决定：只用 depth，不用 RGB

paper Appendix G 给了一个很 engineering 的理由：洗衣机里光照极差，sim 的 RGB 和 real 根本对不上，RGB gap 太大。

他们选了"几何 only"的保守路线，相当于把 sim-to-real gap 限制在 depth sensor noise 这一个维度。Kinect 的 depth noise 已经被 previous garment 工作 (UniGarmentManip, Where2Explore) 验证过 acceptable。

反例 Figure 22 / 23 挺有意思：两个 clutter RGB 几乎一样但 point cloud 不同（因为褶皱 depth 不同），affordance 能区分；反之两个 clutter point cloud 几乎一样但 RGB 不同（因为颜色相似），affordance 就分不开。这其实是个 limitation，但 paper 论证了大部分场景 depth 足够。

---

## Baseline 失败的 takeaway

Table 2 里有一个很 striking 的不对称：Support-M 在 Sofa 上 success 0.784（还行），在 WashingMachine 上 0.562（惨败）。这告诉我们 Sofa 是"简单场景"——开放空间，garment 形状完整，symbolic segmentation 还能 work；但 WM 和 Basket 这种深度 occlusion 场景才是真 cluttered deformable，symbolic 路线立即崩。

Ours 在三个场景都 ~0.8，说明 point-level affordance 对场景类型 robust，因为是 geometric-driven 而不是 semantic-driven。

GPT-4o 在所有场景都 ~0.4，inference 时间 8 秒（affordance < 0.1 秒）。这印证了 VLM 在 deformable + heavy occlusion 场景的 geometric precision 严重不足，不是 prompt 能救的。

---

## 直觉压缩成三句话

1. **回避 segmentation**：直接在 raw point cloud 上学 per-point actionability，让 backbone 自己 implicit 学到 garment boundary。symbolic 路线在 deformable clutter 下会塌。
2. **cascade distillation**：simulator 当 oracle，retrieval model 学"现在能不能抓"，place model 学"pick-place 之后能不能抓"，pick model 学"哪个点 pick 后能 place 出好结果"。这是 supervised 版的 value iteration。
3. **threshold trigger adaptation**：$P_{high} < 0.1$ 就先 reorganize 再抓。random stirring 没用，必须 affordance-guided。

---

## 在 landscape 中的位置

```
1D rope (SoftGym)
  ↓
2D fabric (FabricFlowNet, SpeedFolding)
  ↓
single 3D garment (UniGarmentManip)
  ↓
cluttered 3D garments (this paper)
  ↓
(未来) long-horizon household laundry 全流程
```

每一步加一个 dimension：rope→fabric 是 topology 升维，fabric→garment 是 shape 复杂化，single→cluttered 是 inter-object relation。

GarmentPile 的贡献是把"deformable clutter"这个 dimension 用 point-level affordance 拍平，回避了 segmentation 这个 ill-posed 中间步骤。

---

## 我的 take

这个 paper 最 elegant 的地方是 cascade supervision——用 simulator 当 oracle，把 RL 的 value iteration 用 supervised learning 实现了。这种范式在 deformable manipulation 里应该能推广到更 long-horizon 的 task，比如 fold-then-stack，或者 sort-then-hang。

最弱的点是 $P_{high} > 0.1$ 这个 trigger threshold 没有 sensitivity analysis，distribution shift 下可能不 robust。

最有想象力的 follow-up 是把 adaptation 的 pick-place primitive 换成 diffusion policy，让 reorganization action 连续化，可能能处理更细的 reorganization 需求（比如轻轻拨一下袖子）。

---

Reference:
- Paper: https://garmentpile.github.io/
- GarmentLab env: https://garmentlab.github.io/
- PointNet++: https://github.com/charlesq34/pointnet2
- UniGarmentManip: https://github.com/RvIndustryCoding/UniGarmentManip
- SpeedFolding: https://sites.google.com/berkeley.edu/speedfolding
- SoftGym: https://sites.google.com/view/softgym
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Where2Act: https://github.com/daerriemooo/where2act
- DAgger: https://arxiv.org/abs/1011.0686

---

# GarmentPile 深度技术解读

Andrej 你好，这篇 paper 来自 PKU 的 Hao Dong 组（CFCS），是 GarmentLab (NeurIPS 2024) 之后的延续工作。它把 deformable manipulation 从 single-garment 推进到 cluttered multi-garment retrieval，核心贡献是用 point-level affordance + affordance-guided adaptation 来处理高度纠缠的衣物堆。项目页：https://garmentpile.github.io/ ；GarmentLab 项目页：https://garmentlab.github.io/ 。

下面我会从 motivation、formulation、cascade supervision 设计、PointNet++ 层次聚合直觉、adaptation 触发判据、online data 训练、sim-to-real 取舍、baseline 失败诊断、实验数据直觉，逐层拆开。

---

## 1. 为什么 cluttered garment 是一个新问题

previous works 主要分两类：
1. **single-garment manipulation**：unfold (Flingbot, Ha & Song, CoRL 2022, https://flingbot.github.io/ )、fold (SpeedFolding, Avigal et al., IROS 2022, https://sites.google.com/berkeley.edu/speedfolding )、hang (UniGarmentManip, Wu et al., CVPR 2024, https://github.com/RvIndustryCoding/UniGarmentManip )、dressing-up (One Policy to Dress Them All, Wang et al., RSS 2023, https://dressing-policy.github.io/ )。这类工作把 high-dim state space 归结为 single object 的状态估计 + dense correspondence。
2. **cluttered rigid object manipulation**：graspness (Wang et al., ICCV 2021)、Contact-GraspNet (Sundermeyer et al., ICRA 2021, https://github.com/NVlabs/contact_graspnet )、retrieval affordance (Li et al., RSS 2024)、Mechanical Search (Kurenkov et al., IROS 2020)。这类假设 object 之间刚性、可分割、有清晰 boundary。

garment clutter 的特殊性：
- **state space 几乎无穷**：同一件 shirt 可以有无数种褶皱构型；多件叠加 → 状态爆炸。
- **没有 boundary**：SAM (Kirillov et al., https://arxiv.org/abs/2304.02643 ) 在 washing machine / basket 这种深度 occlusion 下经常把"半件衣服"误分割成"一整件"（paper Appendix A, Figure 6, 12）。
- **relations 非 rigid**：garment A 压在 B 上不等于 B 不能动，因为 B 可能从某个角被抽出而不影响 A；但 A 的袖子穿进 B 的领口就是强 coupling。
- **failure mode 软**：rigid 场景抓错要么撞要么失败，garment 场景抓错可能"目标落地"或"把邻居拖出来"，这俩都是 partial failure，policy 很难 self-correct。

paper 提的两个 failure：
- target contacts floor (uncleanliness)
- neighbors dragged out (unsafety/entanglement)

这其实是把"成功"定义成多约束的复合条件，而不是"是否抓起来"。

---

## 2. Problem Formulation 细节

输入：$O \in \mathbb{R}^{N \times 3}$，N 个点的 raw 3D point cloud（**没有 RGB**，这点后面会讲为什么）。

输出 action：
- **retrieval action**：单一 grasp point $p_{retrieve} \in \mathbb{R}^3$ + heuristic orientation（简化掉 orientation learning）
- **adaptation action**：pick-place pair $(p_{pick}, p_{place}) \in \mathbb{R}^3 \times \mathbb{R}^3$，用来 reorganize scene

per-point affordance maps：
$$A^{retrieve}, A^{pick}, A_{p_{pick}}^{place} \in \mathbb{R}^{N}$$

注意 $A^{place}$ 的下标 $p_{pick}$ —— place affordance 是**条件 affordance**，给定 pick point 之后才有意义。这是一个很重要的 disentanglement：直接学 $A(p_{pick}, p_{place})$ 的联合 actionability 空间是 $\mathbb{R}^{N \times N}$，对 N≈几千的点云学不动；条件化之后变成两次 $\mathbb{R}^N$ 的预测，可以级联。

---

## 3. 三个 module 的 cascade supervision —— paper 最关键的设计

paper 训练顺序是 **Retrieval → Place → Pick**，每一级用上一级的 output 当监督。这其实是 imitation learning 中的一种 self-supervised curriculum，类似 DAgger 但用 model 当 oracle。

### 3.1 Retrieval Affordance $\mathcal{M}_{retrieve}$

architecture：PointNet++ backbone $\mathbf{F}_{retrieve}$ → per-point feature $f_p^{retrieve} \in \mathbb{R}^{128}$ → MLP + sigmoid → $\hat{g}_p^{retrieve} \in [0,1]$。

ground truth 来源：在 simulator 里**直接执行** retrieval action on point p，看结果。$g_p^{retrieve} = 1$ 当 success，0 当 failure。这是一个最干净的 self-supervised label —— 没有 human annotation，全靠物理仿真判定。

loss（公式 1）：
$$L_{retrieve} = -\Big( g_p^{retrieve} \cdot \log \hat{g}_p^{retrieve} + (1-g_p^{retrieve}) \cdot \log(1-\hat{g}_p^{retrieve}) \Big)$$

变量含义：
- $g_p^{retrieve} \in \{0,1\}$：在点 p 处执行 retrieval 是否成功（成功 = 既不落地也不拖累邻居）
- $\hat{g}_p^{retrieve} \in [0,1]$：模型预测的 actionability score
- 公式就是标准 binary cross entropy，sigmoid 输出

数据：20,000 scenes, 120 epochs, batch 128, 4090 GPU, <24h。

### 3.2 Place Affordance $\mathcal{M}_{place}$ —— 条件 affordance 的核心

为什么先训练 place 而不是先训练 pick？这是 paper 4.3 节最 subtle 的设计：

- **pick point 没有 direct feedback**：你 pick 一个点，后面 place 千千万万，无法判定这个 pick 本身好不好
- **place point 有 direct feedback**：给定已经 pick 起来的 $p_{pick}$，放到候选 place p 后，scene 立刻变成新状态 $O'$，可以用 trained retrieval affordance 评估 $O'$ 的 actionability 提升

所以 place 是"低成本、高信息量"的中间层。

architecture（Figure 3 lower-right）：
- 两个 PointNet++：$\mathbf{F}_{place}^1$ 提取 $p_{pick}$ 的 feature $f_{p_{pick}}^{place_1}$，$\mathbf{F}_{place}^2$ 提取候选 place 点 p 的 feature $f_p^{place_2}$
- concat → MLP + sigmoid → $\hat{g}_{p|p_{pick}}^{place} \in [0,1]$

ground truth：
- 执行 pick-place $(p_{pick}, p)$ → 得到 $O'$
- 用 trained $\mathcal{M}_{retrieve}$ 推断 $A^{retrieve}(O')$
- 若 $A^{retrieve}(O')$ 相对初始 $A^{retrieve}(O)$ 提升超过 margin → $g_{p|p_{pick}}^{place} = 1$
- 否则 0

loss（公式 2）：
$$L_{place} = -\Big( g_{p|p_{pick}}^{place} \log \hat{g}_{p|p_{pick}}^{place} + (1-g_{p|p_{pick}}^{place}) \log(1-\hat{g}_{p|p_{pick}}^{place}) \Big)$$

变量：
- $g_{p|p_{pick}}^{place}$：给定 pick 点，把衣物放到 p 是否能让 retrieval affordance 改善
- $\hat{g}_{p|p_{pick}}^{place}$：模型预测
- margin 在 paper 里没给具体值，但 implicitly 是 $P_{high}(O') - P_{high}(O) > \epsilon$

这个设计的妙处：**retrieval model 的"好坏判断"被蒸馏到 place model 里**。Retrieval model 看的是"现在能不能成功抓"，place model 学的是"经过这次 pick-place 之后能不能成功抓"，等价于在学一个 1-step lookahead value function。

### 3.3 Pick Affordance $\mathcal{M}_{pick}$ —— 用 place model 反向监督

训练顺序最后一步。给定 pick point p，用 trained place module $\mathcal{M}_{place}$ 找出最优 $p_{place}^{*}$，然后执行 (p, $p_{place}^{*}$)，看 retrieval affordance 是否改善。这相当于把 place model 当作"oracle policy"给 pick 提供 label。

architecture：单 PointNet++ $\mathbf{F}_{pick}$ → $f_p^{pick} \in \mathbb{R}^{128}$ → MLP + sigmoid → $\hat{g}_p^{pick}$。

loss（公式 3）：
$$L_{pick} = -\Big( g_p^{pick} \log \hat{g}_p^{pick} + (1-g_p^{pick}) \log(1-\hat{g}_p^{pick}) \Big)$$

变量含义：
- $g_p^{pick} \in \{0,1\}$：pick 点 p 之后，用最优 place，retrieval affordance 是否提升
- $\hat{g}_p^{pick} \in [0,1]$：模型预测

8,000 数据, 80 epochs, batch 64（因为有两个 PointNet++，显存大）。

### 3.4 Cascade 的直觉

可以把它看成一个 implicit 的 2-step value iteration：
- $V_{retrieve}(O) = \max_p A^{retrieve}_p(O)$ —— 当前最优抓取点的 affordance
- $Q_{place}(O, p_{pick}, p_{place}) = V_{retrieve}(O')$ —— pick-place 之后的状态值
- $V_{place}(O, p_{pick}) = \max_{p_{place}} Q_{place}$ —— 最优 place 的值
- $Q_{pick}(O, p) = V_{place}(O, p)$ —— pick p 之后能得到的最好后续值
- $A^{pick}_p \propto Q_{pick}(O, p) - V_{retrieve}(O)$ —— advantage

paper 没显式写 advantage，但 ground truth 的 "margin" 判定其实就是 advantage > 0。这跟 reinforcement learning 的 Q-learning / actor-critic 框架在精神上是一致的，只是用 supervised BCE 替代了 TD error。

参考类似思路：
- Where2Act (Mo et al., CVPR 2021, https://github.com/daerduomaaa/where2act ) 在 articulated object 上也是 per-point affordance + sampled actions
- AdaAfford (Wang et al., ECCV 2022, https://github.com/Jiaqian94/AdaAfford ) 是 few-shot 修正 affordance
- DualAfford (Zhao et al., ICLR 2023, https://github.com/yaoyz96/DualAfford ) 双臂 affordance
- General Flow (Yuan et al., https://arxiv.org/abs/2401.11439 ) 把 affordance 抽象成 general flow

---

## 4. PointNet++ backbone 的层次聚合直觉

paper 在 Section 4.2 和 Figure 3 upper-right 强调：per-point feature 同时聚合三层信息：

1. **Local geometry**：抓握点的局部曲率、褶皱走向，决定 gripper 能不能稳定夹住一片布（vs 滑掉）
2. **Global structure**：整件衣物的整体形状，决定"抓这个点之后衣物的其他部分会不会落地"——比如抓 corner 容易让对侧落地，抓 middle 更稳
3. **Inter-object relation**：当前点所属衣物和其他衣物的几何关系，决定"拉这个点会不会把压在下面的衣物一起带出来"

PointNet++ 的 set abstraction layer 天然给出这种 local→global 的层次特征：
- SA1 (N×64)：local 邻域几何（半径 r1）
- SA2 (N×128)：稍大邻域，开始捕获 structural 信息
- SA3 (N×128)：全局 context，relation 信息

paper 没有改 PointNet++ 架构本身，关键是用 simulator 自动生成"哪些点真的能成功"的 label，让 backbone 自己学到这三层 aggregation 都对最终 success 有贡献。这种"不设计 explicit structure、靠监督信号涌现出 structure-aware feature"的做法，跟 Neural Radiance Field 类似——结构由 loss 塑造。

可视化在 Figure 4 / 5 / 8 里很清楚：
- 同件衣物的 middle region 高亮、corner 暗淡 → structure-aware
- 上层 garment 全高亮、下层 garment 暗淡 → relation-aware
- 两件衣物交界处低分 → 几何上避免多件同时被抓

---

## 5. Adaptation module 的触发判据

paper Section 4.3 给了一个非常 empirical 的 trigger：

$$P_{high} = \frac{|\{p : A_p^{retrieve} > 0.9\}|}{N}$$

当 $P_{high} < 0.1$ 时触发 adaptation（pick-place），直到 $P_{high} > 0.1$ 为止。

这个 0.1 是经验值，paper 说"observed that the success rate is significantly high when $P_{high} > 0.1$"。Table 5 给了 adaptation rounds vs success rate：

| Rounds | 0 | 1 | 2 | 3 | 3-rand |
|---|---|---|---|---|---|
| Success | 0.712 | 0.782 | 0.803 | 0.805 | 0.719 |

直觉读这张表：
- 0 rounds（pure retrieval affordance）：0.712，说明 retrieval model 在 tangled 场景仍有 28% failure
- 1 round adaptation 提升 +7%，2 rounds +9.1%，3 rounds +9.3% —— 边际递减，2 rounds 之后基本饱和
- 3-rand（随机 adaptation 3 次）只有 0.719，几乎等于 0 rounds，证明 **adaptation 的"智能"全在 pick & place affordance 上**，不是动作本身带来的随机性扰动

这点很重要：random stirring 不解决问题，必须有目的性的 reorganization。

---

## 6. Online data 训练策略

paper Section 4.4 的 online data boosting 是一个简化版 DAgger：

1. 用 offline-trained model 在随机 sampled scenes 上推断 $p_{retrieve}$
2. 实际执行，若失败 → 加入 mistake buffer
3. buffer 满到 64 → 训练 batch = 64 online + 64 offline（保持旧知识）
4. 迭代直到 performance variance 收敛

为什么这个有效：garment clutter 的 state space 太大，offline random sampling 无法覆盖 long-tail failure modes。online mistake 数据相当于 importance sampling 失败分布。

这套做法跟：
- DAgger (Ross et al., 2011, https://arxiv.org/abs/1011.0686 )
- AdaAfford (Wang et al., ECCV 2022, https://arxiv.org/abs/2207.10361 ) 的 few-shot affordance adaptation
- Where2Explore (Ning et al., https://arxiv.org/abs/2309.07473 ) 的 few-shot novel category exploration

思路一脉相承。

---

## 7. Sim-to-real 取舍：为什么不用 RGB

paper Appendix G 给了一个非常 engineering 的回答：

> "there is a significant gap in color information between simulation and reality, particularly in low-light scenes like washing machine."

具体失败模式：
- 洗衣机内光照极差 → sim 里渲染的颜色和 real 完全不同
- 类似颜色衣物在 RGB 上无法区分，但 point cloud 的 depth 能区分褶皱
- Figure 22 / 23 给了反例：相似 RGB 但不同 point cloud，affordance 能正确区分

这跟 NVIDIA 的 BEHAVIOR / Habitat / OmniGibson 这些 sim-to-real 工作的 RGB-domain-randomization 路线相反。garment pile 工作选了"几何 only"的更保守路线，相当于把 sim-to-real gap 限制在 depth sensor noise（Kinect）这一维度上。

参考类似思路：
- Where2Act 也是 raw point cloud only
- UniGarmentManip 用 dense correspondence，也是 depth-based
- 反例：GPT-Fabric-M 用 RGB → 表 2 显示 0.4 量级 success rate，惨败

---

## 8. Baseline 失败原因深度分析

### 8.1 Where2Act (Mo et al., CVPR 2021, https://github.com/daerriemooo/where2act )

sim success: 0.585 / 0.643 / 0.624
real success: 9/15 / 10/15 / 8/15

Where2Act 是 articulated object 上的 per-point affordance + primitive (pull / push) sampling。它失败因为：
- primitive 是 pull/push，没有 pick-place adaptation 的概念
- 它对单件 articulated object 的 joint 关系建模很强，但 garment 没有 joint
- per-point score 预测的是"primitive 执行后该点是否动"，对 retrieval 这种 long-horizon success 判定不够

### 8.2 Support-M (modified from Supporting Relation, Kirillov et al. SAM-based)

sim success: 0.562 / 0.784 / 0.684
real success: 8/15 / 12/15 / 9/15

Support-M 思路：用 SAM 分割每件衣物，然后找"不被其他衣物支撑"的那件去抓。

- Sofa 上表现最好（0.784）：因为 sofa 开放空间，occlusion 轻，SAM 能分割
- WashingMachine / Basket 上惨败（0.562 / 0.684）：因为深度 occlusion + 严重形变，SAM 把"半件衣服"当"一整件"

paper Appendix B 还 fine-tune 了 SAM，sim 提升到 0.67，real 依然 8/15。原因有二：
- 即使分割对，retrieval 需要选 specific point（不是整件都能抓），point-level representation 才能学
- sim-real image gap

这其实揭示了一个 general 教训：**分割 + relation reasoning 这条 symbolic 路线在 deformable clutter 下会塌掉**，因为分割本身在 deformable + occluded 场景是 ill-posed 的（什么算一件？半截袖子算吗？）。

### 8.3 GPT-Fabric-M (modified from GPT-Fabric, Raval et al., https://arxiv.org/abs/2406.09640 )

sim success: 0.463 / 0.408 / 0.384
real success: 6/15 / 7/15 / 6/15

GPT-4o prompt（Appendix C 给了完整 prompt）让它输出 optimal grabbing pixel 坐标。结果：
- 在复杂 clutter 下 GPT-4o 实际上"随机选点"
- 推理时间 8s vs affordance < 0.1s

这印证了一个 trend：VLM 在 rigid clutter 上还行，在 deformable + 高 occlusion 下几何理解崩溃。原因可能是训练数据里这类场景少 + 像素级 precision 不是 VLM 强项。

paper Appendix C 的 prompt 是这样的（节选）：
> "Provide me with the optimal grabbing point as precisely as possible, which should be the coordinates of a pixel in the RGB image."

这种 prompt 让 GPT-4o 输出 (201, 313) 这种具体像素坐标。它的失败 case 在 Figure 13-15 显示得很清楚。

---

## 9. 实验数据直觉

### 9.1 主表 (Table 2) 的不对称性

| Method | WM | Sofa | Basket |
|---|---|---|---|
| Where2Act | 0.585 | 0.643 | 0.624 |
| Support-M | 0.562 | **0.784** | 0.684 |
| GPT-Fabric | 0.463 | 0.408 | 0.384 |
| Ours | **0.805** | **0.819** | **0.792** |

Support-M 在 Sofa 上突然高（0.784），其他场景塌掉。这告诉我们：**Sofa 是"简单"场景**——开放空间，garment 完整可分割，所以 symbolic 路线 work；但 WM 和 Basket 才是真正挑战 cluttered deformable manipulation 的场景，symbolic 路线立即崩。

Ours 在三个场景都 ~0.8，说明 point-level affordance 对场景类型 robust，几何驱动而非语义驱动。

### 9.2 Ablation (Table 1) 的 hierarchy

| Variant | WM | Sofa | Basket |
|---|---|---|---|
| w/o Adaptation | 0.712 | 0.702 | 0.693 |
| w/o Pick Afford | 0.724 | 0.704 | 0.716 |
| w/o Place Afford | 0.778 | 0.743 | 0.731 |
| Full | 0.805 | 0.819 | 0.792 |

读法：
- 去掉 adaptation 整体 -10%，证明 adaptation 是核心贡献
- w/o Pick Afford 几乎 = w/o Adaptation（+1% 左右），说明 **pick affordance 是 adaptation 的真正大脑**：随便选个 pick 点再好的 place 也救不回来
- w/o Place Afford 比 w/o Pick Afford 高 +5%，说明 place affordance 贡献小于 pick affordance，但仍显著

直觉解释：garment 的"哪个点被抓起来能改善场景"是高度 sparse 的——大部分点抓起来反而让场景更乱。而 place 的随机性相对没那么致命，因为放到大部分位置都不会比原来更糟。所以 pick 的 affordance 信号更 critical。

### 9.3 Adaptation rounds (Table 5) 的边际递减

| Rounds | 0 | 1 | 2 | 3 | 3-rand |
|---|---|---|---|---|---|
| Succ | 0.712 | 0.782 | 0.803 | 0.805 | 0.719 |

- 第 1 round +7%，第 2 round +2%，第 3 round +0.2% → 大部分 tangled scene 只需 1-2 次 pick-place 就被 unblock
- 3-rand 跟 0 round 几乎一样（0.719 vs 0.712）→ random stirring 完全无效，必须 affordance-guided

这印证了 retrieval affordance 的 trigger 阈值 $P_{high} > 0.1$ 是 reasonable 的——大部分场景 1-2 次 adaptation 就能跨过这个阈值。

---

## 10. Generalization 数据

paper Appendix H 给：
- seen shapes: 0.805
- novel shapes in seen categories: 0.754
- novel categories: 0.725

掉 ~8% 跨 novel categories，对 deformable object 来说算相当 robust 了。原因 paper 解释为：garment geometry / structure / relation 三层信号是 cross-category invariant 的（褶皱、整体形状、堆叠关系这些 feature 跟 category 无关）。

Figure 24 展示了 scarf / sock / hat 这种 novel category 上 affordance map 仍合理。

---

## 11. Limitations

paper Section 6 / Appendix I 自承：
- 无法模拟 garment knots（衣物打结）—— Isaac Sim 的 cloth solver 不支持这种拓扑约束
- 只用 parallel gripper，没有 dexterous hand / bimanual
- 没用 RGB 信息（虽然 Figure 22-23 论证了 depth 足够，但 RGB 对颜色相似场景理论上仍有用）

未来方向可能：
- bimanual + dexterous（参考 SpeedFolding 的双臂 + Dex-Net 思路）
- 加入 RGB 但用 domain randomization 弥合 sim-real gap（参考 IsaacGym / IsaacLab 的 lighting randomization）
- 用 diffusion policy 替代 pick-place primitive（参考 Diffusion Policy, Chi et al., RSS 2023, https://diffusion-policy.cs.columbia.edu/ ），让 action space 更连续

---

## 12. 在 deformable manipulation landscape 中的位置

把这篇 paper 放在整个 deformable manipulation 的 evolution 里看：

```
1D rope/cable (SoftGym, Lin et al. CoRL 2020, https://sites.google.com/view/softgym )
   ↓
2D fabric (FabricFlowNet, Weng et al., CoRL 2022; SpeedFolding)
   ↓
single 3D garment (UniGarmentManip, CVPR 2024)
   ↓
cluttered 3D garments (GarmentPile, this paper)
   ↓
(未来) long-horizon household laundry task（fold + sort + hang 全流程）
```

每一步增加一个 dimension of complexity：
- rope → fabric：从 1D topology 到 2D
- fabric → garment：从 regular shape 到 irregular category-specific shape
- single → cluttered：从 isolated object 到 inter-object relations

GarmentPile 的 point-level affordance 是处理 cluttered deformable 的一个 elegant 抽象，因为它把"分割"这个 ill-posed 问题回避掉了——直接在 raw point cloud 上预测 per-point actionability，让 backbone 自己隐式学到 garment boundary。

---

## 13. 我会问的问题 / 可能的 follow-up

1. **$P_{high} > 0.1$ 这个阈值在 distribution shift 下是否 robust？** 不同 garment 类别、不同 clutteredness 下应该不一样。paper 没给 sensitivity analysis。
2. **adaptation 会不会"越 adapt 越乱"？** paper 的 margin 判定要求 improvement，但如果 model 误判呢？没看到 negative adaptation 的 case study。
3. **orientation learning 缺失**：retrieval 用 heuristic orientation，对长形 garment（scarf, trousers）应该有影响。这个 simplification 是不是限制了上限？
4. **point cloud 视角单一**：washing machine 是 front view，sofa 是 top-down。多视角 fusion 会不会更好？参考 VGN (Breyer et al., CoRL 2021, https://github.com/ethz-asl/volumetric_grasping_network )。
5. **Diffusion Policy 整合**：把 adaptation 的 pick-place 换成 diffusion policy 学习连续 action distribution，可能能处理更细的 reorganization。
6. **Language conditioning**：现在 retrieval 是"任意一件"，加 language "give me the red shirt" 会更接近家用场景。参考 CLIPort (Shridhar et al., CoRL 2021, https://cliport.com/ )、VLM grasping (VL-Grasp, Lu et al., IROS 2023)。
7. **knot 场景**：sim 不支持，但 real world 衣物打结很常见。可能需要 cable manipulation 的方法（AdaptiGraph, Zhang et al., RSS 2024, https://adaptigraph.github.io/ ）扩展到 surface deformable。

---

## 14. 总结 intuition

GarmentPile 的核心 intuition 可以压缩成三句话：

1. **把"哪些点能抓"建模为 per-point regression**，让 backbone 自己涌现 garment geometry / structure / relation 三层 feature，而不是显式分割 + 关系推理（这条路在 deformable clutter 下死掉）。
2. **把"先扒拉再抓"建模为 pick-place cascade**，用 retrieval affordance 当 1-step lookahead value function 反向蒸馏 place 和 pick，避免学 (pick, place) 联合空间。
3. **用 $P_{high}$ 阈值 trigger adaptation**，相当于一个 implicit critic 来判断 scene 是否 manipulatable，让 policy 在 failure mode 上 self-correct 而不是硬抓。

整个 framework 在精神上很像 value iteration 的 supervised 版本：每级 model 是一级 Q function，ground truth 来自 simulator 而不是 Bellman backup。这种"用 simulator 当 oracle + cascade distillation"的范式在 deformable manipulation 里应该可以推广到其他 long-horizon task（比如 fold-then-stack）。

---

## Reference Links

- Paper project page: https://garmentpile.github.io/
- GarmentLab (environment): https://garmentlab.github.io/ , https://arxiv.org/abs/2411.13707
- ClothesNet (assets): https://github.com/OpenRobotLiabrary/ClothesNet
- PointNet++: https://github.com/charlesq34/pointnet2
- Where2Act: https://github.com/daerriemooo/where2act
- AdaAfford: https://github.com/Jiaqian94/AdaAfford
- UniGarmentManip: https://github.com/RvIndustryCoding/UniGarmentManip
- FlingBot: https://flingbot.github.io/
- SpeedFolding: https://sites.google.com/berkeley.edu/speedfolding
- SoftGym: https://sites.google.com/view/softgym
- SAM: https://arxiv.org/abs/2304.02643
- GPT-Fabric: https://arxiv.org/abs/2406.09640
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- CLIPort: https://cliport.com/
- AdaptiGraph: https://adaptigraph.github.io/
- VGN: https://github.com/ethz-asl/volumetric_grasping_network
- Contact-GraspNet: https://github.com/NVlabs/contact_graspnet
- Where2Explore: https://arxiv.org/abs/2309.07473

希望这个拆解对你 build intuition 有帮助，Andrej。如果想再 deep dive 任何一个 module（比如 PointNet++ 在 garment 上的 feature emergence、cascade supervision 的 gradient flow、或者 simulator 的 cloth physics setup），可以继续问。
