---
source_pdf: JANUSVLN.pdf
paper_sha256: 2fa6fd87a1e72551202f00b799afd06d8755121dd938f9ae1e247a27ec9f0b69
processed_at: '2026-08-05T10:41:20-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# JanusVLN 用人话讲

## 一句话先说清楚

VLN agent 之前的"记忆"要么存成文字地图（空间信息全丢），要么存所有历史帧（算不动），JanusVLN 借鉴人脑左右脑分工，用两个 encoder 分别抓"这是啥"和"在哪啥姿势"，只存它们吐出的 KV cache（固定大小，不增长），结果又快又准。

参考项目主页：https://miv-xjtu.github.io/JanusVLN.github.io/

---

## 1. 这 paper 要解决的痛点

VLN agent 边走边看 RGB 视频，听指令导航。走到第 100 步时，前 99 步看到的东西怎么处理？

**方案 A：转成文字描述存起来**（MapNav 这么干）
- "左边有桌子，前面有门"
- 问题：文字表达不了"门在桌子左前方 2 米、朝东"这种空间关系
- 越存越啰嗦，重复描述一堆冗余

**方案 B：把所有历史帧都存下来**（Uni-NaVid、StreamVLN 这么干）
- 每次预测下一步重新跑一遍所有历史帧的 encoder
- 问题：算不动。48G GPU 在 48 帧就 OOM（看 Figure 3）

**方案 C：上 depth sensor / 点云**（g3D-LF、NaVid-4D 这么干）
- 问题：真实机器人没这硬件

还有个更隐蔽的问题：**所有这些方法的 visual encoder 都是 CLIP paradigm 训的，只会读 2D 语义**。可导航本质是 3D 任务——你得知道走廊多深、门口在哪、转弯多少度。一张 2D RGB 其实包含 3D cues（透视、遮挡、几何结构），人一眼能看出深度，模型却视而不见。

参考 VLN-CE benchmark：https://jacobkrantz.github.io/vln-ce/

---

## 2. JanusVLN 的核心 idea

借鉴人脑半球分工（Gazzaniga 1967 的 split-brain 实验）：
- 左脑管语义（"what"）
- 右脑管空间（"where"）

所以搞两个 encoder：

| Encoder | 干啥 | 来源 |
|---|---|---|
| Semantic encoder | 2D 图像理解"这是桌子、那是椅子" | Qwen2.5-VL 自带的 |
| Spatial encoder | 从纯 RGB 推 3D 几何 | VGGT（CVPR 2025） |

VGGT 训练数据是"图片-3D 点云"对，所以内置 3D prior。给它一张 RGB 就吐出 geometric tokens，里面编码深度、结构信息——**不需要真的 depth sensor**。

参考：
- Gazzaniga split-brain: https://www.scientificamerican.com/article/the-split-brain-in-man/
- VGGT: https://vggt.github.io/
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923

---

## 3. 最巧妙的 trick：把 KV cache 当 memory

不存原帧，存 encoder 处理过的 KV。

Transformer 的 attention 里每个 token 算 K（key）和 V（value）。过去帧算过的 KV 直接缓存，新帧来了做 cross-attention 取这些 KV——相当于"我已经想清楚过去看到啥了，新画面直接和我的记忆对话"。

公式 (2)：

$$G_t = \text{Decoder}(\text{CrossAttn}(\text{Encoder}(x_t), \{M_{initial}, M_{sliding}\}))$$

- $x_t$：当前新帧
- $M_{initial}$：起点帧的 KV cache（永久保留）
- $M_{sliding}$：最近 $n$ 帧的 KV cache（FIFO 滚动）
- $G_t$：当前帧与历史 memory 交互后的 geometric tokens

### 但一直存所有 KV 也会爆，所以搞了 hybrid 策略

**Initial window**（8 帧）：起点帧的 KV 永久保留
- 起点是整个导航的"坐标系锚点"
- agent 在长程导航中要反复参考起点上下文

**Sliding window**（48 帧）：最近 48 帧的 KV，FIFO 滚动
- 聚焦当前 context，决策只关心最近发生啥

总大小固定 56 帧 KV，不管走 100 步还是 1000 步都一样。

### 这个 initial window 的灵感来自 StreamingLLM 的 attention sink

LLM 对序列开头几个 token 有异常高的 attention，丢掉性能就崩。VLN 里起点帧扮演同样角色——Table 5 验证：去掉 initial KV，SR 从 52.8 掉到 51.0，SPL 从 49.2 掉到 47.5。

参考 StreamingLLM: https://arxiv.org/abs/2309.17453

---

## 4. 两个 feature 怎么融合

公式 (4)：

$$F_t = S_t' + \lambda \cdot \text{MLP}(G_t')$$

- $S_t'$：semantic feature（spatial merge 后，shape 为 $\lfloor H/2p \rfloor \times \lfloor W/2p \rfloor \times C$）
- $G_t'$：geometric feature（同样 spatial merge 对齐 shape）
- $\lambda$：spatial feature 权重，最佳 0.2
- $F_t$：最终融合 feature，喂给 LLM backbone 预测 action

加法就行，cross-attention 反而略差（Table 6：CrossAttn SR=52.1，加法 SR=52.8）。

直觉：semantic 是主信号（要听懂指令 grounding 到视觉），spatial 是辅助调味，加多了喧宾夺主（λ=0.5 时 SR 掉到 50.4）。

类似 LoRA 的 scaling factor $\alpha/r$，控制 adapter 信号强度。

---

## 5. 为什么又快又好

### 快

VGGT 原生每加一帧重跑整个序列，复杂度 $O(T^2 \cdot d)$。JanusVLN 复用 cached KV，复杂度 $O(T \cdot n \cdot d)$，$n=56$ 固定。

Table 5 实测：

| Memory Size | VGGT 原生 | JanusVLN cached | 加速 |
|---|---|---|---|
| 8 帧 | 268 ms | 82 ms | 3.3× |
| 32 帧 | 1549 ms | 149 ms | 10.4× |
| 48 帧 | OOM on 48G | 195 ms | ∞ |

### 好

R2R-CE Val-Unseen 主表（Table 1）：

| Method | 输入 | SR↑ | SPL↑ | 训练数据 |
|---|---|---|---|---|
| NaVILA | RGB | 54.0 | 49.0 | 13.1M |
| StreamVLN | RGB | 56.9 | 51.9 | 26.3M |
| **JanusVLN** | **RGB** | **60.5** | **56.8** | **10.7M** |
| JanusVLN* (无额外数据) | RGB | 52.8 | 49.2 | 0 |

用更少数据刷出更高分。数据效率算一下：
- StreamVLN: 56.9 / 26330 ≈ 0.00216 SR per K样本
- JanusVLN: 60.5 / 10692 ≈ 0.00567 SR per K样本 → **数据效率高 2.6 倍**

即使零额外数据（JanusVLN*），SR 52.8 也比 StreamVLN* 的 45.5 高 7.3。

### 好在哪——ablation 揭示

最 clean 的 ablation 是 Table 4：换 DINOv2/SigLIP 2（2D encoder）几乎没提升，换 random init VGGT 也没提升，只有 pretrained VGGT 涨 5.8 SR。

| Encoder | SR↑ | SPL↑ |
|---|---|---|
| w/o extra encoder | 47.0 | 40.9 |
| +DINOv2 | 47.5 | 41.5 |
| +SigLIP 2 | 47.9 | 41.9 |
| +VGGT [random init] | 47.2 | 40.8 |
| +VGGT [pretrained] | **52.8** | **49.2** |

证明收益来自 **3D prior**，不是参数变多。

Table 3 验证两类 memory 都不可缺：

| Configuration | SR↑ | Δ SR |
|---|---|---|
| Full JanusVLN | 52.8 | — |
| w/o Spatial Memory | 47.0 | -5.8 |
| w/o Semantic Memory | 45.5 | -7.3 |
| w/o Dual Memory | 24.8 | -28.0 |

两类 memory 一起拿掉直接崩盘 28 SR——说明它们建立的是 **complementary 表征空间**，缺一个 LLM 失去关键 context anchor。

---

## 6. 关键直觉解读

### 为啥 cross-attention with cached KV ≈ full self-attention？

VGGT 原生 multi-frame attention：

$$\text{Attn}(Q_{new}, K_{1:T}, V_{1:T}) = \text{softmax}\left(\frac{Q_{new} K_{1:T}^T}{\sqrt{d}}\right) V_{1:T}$$

新帧 $Q_{new}$ 与所有历史 $K, V$ 算 attention。

JanusVLN 缓存历史 $K, V$ 后，新帧来时只需 cross-attention 与 cached KV 交互——**数学上等价**，工程上避免 re-encoding。

唯一区别：JanusVLN 限制 history 到 (initial + sliding) 而非全部，是近似。但 attention sinks 保留全局信息，性能损失可忽略。

### 为啥 λ=0.2 最优？

semantic feature $S$ 的信息量远大于 spatial feature $G$（Qwen2.5-VL encoder 训练数据规模是 VGGT 的几十倍）。

$\lambda$ 控制 spatial 信号的"音量"：
- $\lambda=0.5$ 太强，淹没 semantic grounding，SR 掉到 50.4
- $\lambda=0.1$ 太弱，3D prior 没充分利用，SR 掉到 50.2
- $\lambda=0.2$ sweet spot，SR 52.8

### Attention sinks 在 VLN 中的特殊意义

NLP 里 attention sinks 是 BOS token 等无信息 token 吸收冗余 attention。VLN 里**初始帧**是 agent 起始位置的全景观察，包含：
- 整个房间 layout 上下文
- 起点相对目标的 global 几何参考
- 指令的 visual grounding 锚点

这些信息在整个 trajectory 中都需反复参考，类似"导航的起点坐标系"。

---

## 7. 我看到的问题与限制

### Memory 固定大小的代价

56 帧 KV。R2R 平均轨迹 ~10-30 步够用。但 RxR 长指令可能数百步，中间帧 KV 被 evict。Table 5 显示 48→64 帧几乎无增益，说明 56 帧够 R2R-CE，但**更长程任务可能饱和**。

### VGGT frozen 的限制

VGGT encoder 保持 frozen，未在 VLN 数据上微调。Table 4 显示 frozen 已带来 5.8 SR 提升，但 end-to-end fine-tune 可能更高。frozen 选择为节省计算和避免 catastrophic forgetting。

### Real-world 评估规模有限

Figure 4 和 Figure 5 的 real-world 实验是 qualitative 的，仅 Unitree Go2 + Insta360 X5 几个案例。没 quantitative real-world benchmark，sim-to-real gap 未系统评估。

### Initial window 8 帧没消融

Table 5 消融了 sliding window 大小（8/32/48/64），但 initial window 大小固定 8 帧没扫。可能不是最优。

---

## 8. 更大的图景

### Memory paradigm 演化

| Paradigm | 代表方法 | Memory 类型 | 大小 | 空间信息 |
|---|---|---|---|---|
| Explicit textual map | MapNav (ACL'25) | Object node + edge text | 增长 | 丢失 |
| Explicit frame history | Uni-NaVid, StreamVLN | Raw frames | 增长 | 2D only |
| Explicit 3D map | g3D-LF (CVPR'25) | 3D feature fields | 增长 | 3D 但需 depth |
| **Implicit dual** | **JanusVLN** | **KV cache (sem+geo)** | **固定** | **3D prior from RGB** |

### 与 LLM serving 技术的关联

KV cache 管理在 LLM serving 早有成熟技术：PagedAttention (vLLM)、Sliding Window Attention (Mistral)、Ring Attention。JanusVLN 的 hybrid cache 与 StreamingLLM 直接对应，但**未引用更广泛的 KV cache 优化文献**——future work 方向。

参考：
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- Mistral sliding window: https://arxiv.org/abs/2310.06825

### 3D foundation model 当 spatial encoder 的 pattern

VGGT 与 DUSt3R (CVPR 2024)、MASt3R 同属 feed-forward 3D reconstruction 家族。JanusVLN 的设计可推广到任何此类 3D foundation model 作为 spatial encoder。

参考 DUSt3R: https://dust3r.europe.naverlabs.com/

### 与 World Model 的关联

NavMorph (ICCV'25) 用 world model，JanusVLN 用 implicit KV memory。两者本质都是**学习 environment dynamics 的 compressed representation**。JanusVLN 的 KV cache 可视为 world model 的 implicit state——下一步 action 预测基于这个 state。

### 推广到通用 VLA

JanusVLN 是 VLN-specialized。当前趋势（OpenVLA、π0、GR-2）是通用 VLA foundation model。dual implicit memory 思想可移植到 manipulation：semantic encoder + spatial encoder + dual KV cache。

参考：
- OpenVLA: https://openvla.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0
- GR-2: https://agilex-robotics.github.io/gr2/

---

## 9. 一句话总结

JanusVLN 把 VLN 的 memory 问题从"存什么"转到了"怎么存"——存 KV cache 而非原帧，用两个解耦 encoder 抓 complementary 信息，固定大小不增长。又快（10× 加速）又好（SOTA SR 60.5），又省数据（2.6× 数据效率）。核心 insight：navigation 需要的是更结构化的 spatial-semantic 解耦表征，3D prior from RGB + 双 implicit KV cache 是这个 insight 的具体实现。

---

## 10. 如果想 build 更深 intuition，建议深挖

1. **VGGT 的 KV cache 实现细节**：fusion decoder 中具体哪些 layer 的 KV 被 cache？attention output 还是 FFN 前的 hidden state？
2. **Attention pattern 可视化**：spatial encoder 和 semantic encoder 的 attention 在不同帧上如何分布？是否真的 spatial 偏 global、semantic 偏 local？
3. **Failure case analysis**：JanusVLN 在什么类型 instruction 上失败？空间关系密集的？长程依赖的？dynamic scene 的？
4. **Cross-dataset generalization**：RxR 多语言指令（英、印、泰）是否对 spatial encoder 公平？多语言 instruction 是否影响 semantic encoder 表征质量？
5. **与 token pruning 结合**：Qwen2.5-VL 的 2×2 spatial merging 是基础 token 压缩。结合 FastV、VTW 等更激进 pruning 可进一步降低 KV cache 大小。参考 https://arxiv.org/abs/2403.06764

---

# JanusVLN 深度技术讲解

## 1. 核心问题与动机

JanusVLN 解决的是 Vision-and-Language Navigation (VLN) 在 continuous environment 下的三个根本痛点：

1. **Spatial information loss**：现有方法用 textual cognitive map 描述空间关系，文本无法精确表达 3D 几何 orientation 和 distance
2. **Computational redundancy**：每来一帧新观测，需重新处理整个历史序列（VGGT 原生设计如此，48G GPU 在 48 帧时 OOM）
3. **Memory bloat**：explicit memory 随 trajectory length 线性增长，长程导航时模型难以从大量 cluttered memory 中提取关键信息

更深层的问题：**navigation 本质是 3D 物理交互，但 visual encoder 继承 CLIP paradigm，仅在 2D image-text pairs 上预训练**。2D image 不是孤立像素平面，是 3D 物理世界的投影，包含 perspective、occlusion、geometric structure 等 3D cues——人类一眼就能感知 depth，但模型忽略了这些 implicit 3D 信息。

## 2. 核心 Idea：Dual Implicit Neural Memory

灵感来自人脑半球分工（Gazzaniga, 1967）：
- **Left hemisphere**：semantic understanding（"what it is"）
- **Right hemisphere**：3D spatial cognition（"where it is and how it's related"）

JanusVLN 把这两类 memory 建模为**fixed-size、compact 的神经表征**——大小不随轨迹长度增长，模拟人脑有限容量的高效记忆。

参考链接：
- 人脑半球分工研究：https://www.scientificamerican.com/article/the-split-brain-in-man/
- 项目主页：https://miv-xjtu.github.io/JanusVLN.github.io/

## 3. 关键技术组件

### 3.1 VGGT（Visual Geometry Grounded Transformer）

VGGT 是 Wang et al. CVPR 2025 提出的 feed-forward 3D reconstruction foundation model，训练数据是 **pixel-3D point cloud pairs**，编码强 3D perception prior。

公式 (1)：

$$
\{G_t\}_{t=1}^{T} = \text{Decoder}(\text{Encoder}(\{x_t\}_{t=1}^{T})), \quad (P_t, C_t) = \text{Head}(G_t)
$$

变量含义：
- $\{x_t\}_{t=1}^{T}$：输入 RGB 视频帧序列，$x_t \in \mathbb{R}^{3 \times H \times W}$
- $G_t \in \mathbb{R}^{\lfloor H/p \rfloor \times \lfloor W/p \rfloor \times C}$：geometric tokens
  - $p$：patch size（VGGT 默认 14）
  - $C$：feature channel
  - $\lfloor \cdot \rfloor$：下取整
- $P_t \in \mathbb{R}^{3 \times H \times W}$：预测的 point map（每个像素一个 3D 坐标）
- $C_t \in \mathbb{R}^{H \times W}$：per-pixel confidence map

JanusVLN 只用 VGGT 的 **encoder + fusion decoder**，不用 prediction head——关注的是"嵌入 3D geometry prior 的 feature"，而非直接 3D 输出。

参考：https://vggt.github.io/

### 3.2 Implicit Neural Representation

关键 insight：缓存**神经网络 attention module 输出的 KV cache**，而非原始帧。这些 KV 是高阶语义抽象和结构化表征，是 condensed knowledge representation。

公式 (2)：spatial memory 通过 cross-attention 检索历史信息：

$$
G_t = \text{Decoder}(\text{CrossAttn}(\text{Encoder}(x_t), \{M_{initial}, M_{sliding}\}))
$$

变量：
- $M_{initial}$：初始帧的 KV cache（永久保留）
- $M_{sliding}$：滑动窗口内最近 $n$ 帧的 KV cache（FIFO 更新）
- $\text{CrossAttn}(\cdot, \cdot)$：当前帧 image tokens 与历史 KV 的交叉注意力

### 3.3 Hybrid Incremental Update（关键设计）

这个设计借鉴了 **StreamingLLM 的 Attention Sinks**（Xiao et al., ICLR 2024）。研究发现：transformer 对序列最初的几个 token 保持极高 attention 权重，丢弃它们会导致性能崩塌，保留它们即可大幅维持性能。

JanusVLN 的策略：
- **Sliding window** $M_{sliding}$：容量 $n$ 帧，FIFO 队列，关注最近 contextual 信息
- **Initial window** $M_{initial}$：永久保留最初几帧的 KV（实验里 8 帧），作为 "attention sinks" 提供全局锚点

固定大小 memory = initial KV + sliding KV，不随 trajectory 长度增长。

参考 StreamingLLM：https://arxiv.org/abs/2309.17453

### 3.4 JanusVLN 完整架构

**双 encoder 解耦视觉感知**：

公式 (3)：2D semantic encoder（来自 Qwen2.5-VL）：

$$
S_t = \text{Encoder}_{\text{sem}}(x_t), \quad S_t \in \mathbb{R}^{\lfloor H/p \rfloor \times \lfloor W/p \rfloor \times C}
$$

Qwen2.5-VL 用 spatial merging：相邻 $2\times2$ patch 合为一个 token：

$$
S_t' \in \mathbb{R}^{\lfloor H/2p \rfloor \times \lfloor W/2p \rfloor \times C}
$$

公式 (4)：spatial-aware feature fusion：

$$
F_t = S_t' + \lambda \cdot \text{MLP}(G_t')
$$

变量：
- $G_t'$：spatial-geometric feature 经 spatial merging 后（与 $S_t'$ shape 对齐）
- $\lambda$：spatial feature 权重（消融实验最佳值 0.2）
- $F_t$：最终空间-语义增强 visual features
- $\text{MLP}(\cdot)$：两层 lightweight projection

最终把 $F_t$ + instruction embedding $\mathcal{T}$ 喂入 MLLM backbone，预测下一个 low-level action。

参考 Qwen2.5-VL：https://arxiv.org/abs/2502.13923

## 4. Architecture 图解析（Figure 2）

```
RGB video stream → ┌─────────────────┬─────────────────┐
                   ↓                  ↓                 
            Semantic Encoder   Spatial-Geo Encoder (VGGT)
            (Qwen2.5-VL)       (frozen, 3D prior)
                   ↓                  ↓
              KV cache            KV cache
              ↓    ↓              ↓    ↓
        M_initial  M_sliding   M_initial  M_sliding
        (8 frames) (48 frames) (8 frames) (48 frames)
              ↓    ↓              ↓    ↓
              cross-attn with new frame tokens
                   ↓                  ↓
              S_t' (sem tokens)   G_t' (geo tokens)
                   ↓                  ↓
                   └──── MLP fusion (λ=0.2) ──┐
                                             ↓
                                            F_t
                                             ↓
                                  LLM backbone (Qwen2.5-VL 7B)
                                             ↓
                                       Action a_{t+1}
```

## 5. 实验数据深度解读

### 5.1 Table 1：R2R-CE Val-Unseen 主表

最关键的对比：

| Method | Inputs | SR↑ | SPL↑ | External Data |
|---|---|---|---|---|
| NaVILA | RGB | 54.0 | 49.0 | 13132K |
| StreamVLN | RGB | 56.9 | 51.9 | ~26330K |
| JanusVLN | RGB | **60.5** | **56.8** | 10692K |
| JanusVLN* | RGB | 52.8 | 49.2 | 0K |

**数据效率计算**（SR per 1K training samples）：
- StreamVLN：56.9 / 26330 ≈ 0.00216 SR/K
- NaVILA：54.0 / 13132 ≈ 0.00411 SR/K
- JanusVLN：60.5 / 10692 ≈ **0.00567 SR/K**（数据效率高 2.6× vs StreamVLN）

**与多输入方法对比**（即使它们用 pano+odo+depth）：
- g3D-LF（用 depth）：SR=47.2，JanusVLN +12.6
- NaVid-4D（用 depth）：SR=43.8，JanusVLN +16.7
- InstructNav（pano+odo+depth）：SR=24.0，JanusVLN +35.5

### 5.2 Table 3：Dual Memory Ablation（最核心）

| Configuration | NE↓ | OS↑ | SR↑ | SPL↑ | Δ SR |
|---|---|---|---|---|---|
| Full JanusVLN | 5.17 | 58.0 | 52.8 | 49.2 | — |
| w/o Spatial Memory | 6.58 | 54.3 | 47.0 | 40.9 | -5.8 |
| w/o Semantic Memory | 6.75 | 53.1 | 45.5 | 40.0 | -7.3 |
| w/o Dual Memory | 7.85 | 36.9 | 24.8 | 16.8 | -28.0 |

关键直觉：**两类 memory 不是简单 additive，而是 synergistic**。去掉任一个降幅 5.8-7.3 SR，但都去掉暴跌 28.0 SR——说明两者建立的是**complementary 表征空间**，缺失任一个 LLM 失去了关键 context anchor。

### 5.3 Table 4：3D Prior 来源消融（最重要的 ablation 之一）

| Encoder | SR↑ | SPL↑ |
|---|---|---|
| w/o extra encoder | 47.0 | 40.9 |
| +DINOv2 | 47.5 | 41.5 |
| +SigLIP 2 | 47.9 | 41.9 |
| +VGGT [random init] | 47.2 | 40.8 |
| +VGGT [pretrained] | **52.8** | **49.2** |

**关键 insight**：
- DINOv2/SigLIP 2（2D image-text 预训练）几乎无提升——与 Qwen2.5-VL 自带 encoder 信息冗余
- VGGT random init 也无提升——证明收益**不是参数增加**带来的
- 只有 pretrained VGGT 大涨 5.8 SR——证明收益来自 **3D spatial-geometric prior**

这是一个非常 clean 的 ablation，清晰隔离了"3D prior 信息"作为唯一 benefit 来源。

### 5.4 Table 5：推理效率对比（推理实时性关键）

| Memory Size | VGGT Inference | Cached Memory | 加速比 |
|---|---|---|---|
| 8 frames | 268 ms | 82 ms | 3.3× |
| 32 frames | 1549 ms | 149 ms | 10.4× |
| 48 frames | OOM on 48G | 195 ms | ∞ |

VGGT 原生设计每加一帧需 reprocess 整个序列——**二次方复杂度 $O(T^2 \cdot d)$**，原因是 self-attention 跨所有帧计算。

JanusVLN 把复杂度降到 $O(T \cdot n \cdot d)$，$n$ 是固定窗口大小（48）。每帧只需 cross-attention 与固定大小 KV cache 交互。

### 5.5 Table 6：Fusion 策略

| Strategy | SR↑ | SPL↑ |
|---|---|---|
| λ=0.5 | 50.4 | 46.9 |
| λ=0.2 | **52.8** | **49.2** |
| λ=0.1 | 50.2 | 46.6 |
| Concat | 49.4 | 45.7 |
| CrossAttn | 52.1 | 48.6 |

**直觉**：spatial feature 不能压过 semantic（λ=0.5 太高反而下降），因为最终动作预测依赖 instruction grounding，semantic 是主信号。简单 addition 反而优于 cross-attention——可能因为 cross-attention 在 7B scale 上引入额外未训练参数，需更多数据。

### 5.6 Table 7：Data Ablation

| Data | SR↑ | SPL↑ |
|---|---|---|
| w/o Extra Data | 52.8 | 49.2 |
| +ScaleVLN | 55.5 | 50.9 |
| +DAgger | 56.4 | 51.7 |
| +Both | 60.5 | 56.6 |

ScaleVLN 155K trajectories（9207K image-action pairs）+ DAgger 14K trajectories（1485K pairs）。DAgger 比 ScaleVLN 更高效—— DAgger 来自同分布 R2R-CE/RxR-CE 数据，分布对齐更好。

DAgger 算法（Ross et al., 2011 AISTATS）是经典 imitation learning 技术，agent 在执行时收集"专家会采取的动作"做在线纠正。

参考 DAgger：https://arxiv.org/abs/1011.0686

## 6. 与相关工作的位置定位

### 6.1 Memory Paradigm 谱系

| Paradigm | 代表方法 | Memory 类型 | 大小 | 空间信息 |
|---|---|---|---|---|
| Explicit textual map | MapNav (ACL'25) | Object node + edge text | 增长 | 丢失（文本化） |
| Explicit frame history | Uni-NaVid, StreamVLN | Raw frames | 增长 | 2D only |
| Explicit 3D map | g3D-LF (CVPR'25) | 3D feature fields | 增长 | 3D 但需 depth |
| **Implicit dual** | **JanusVLN** | **KV cache (sem+geo)** | **固定** | **3D prior from RGB** |

### 6.2 与 StreamVLN 的对比

StreamVLN (Wei et al., 2025) 也用 KV cache 但只有**单 implicit memory（semantic only）**，没有 spatial-geometric 分支。JanusVLN 借鉴了 StreamVLN 的 slowfast context 建模思路 + DAgger 数据策略，但扩展到 dual encoder + dual memory。

参考 StreamVLN：https://arxiv.org/abs/2507.05240

### 6.3 与 NavMorph 对比

NavMorph (ICCV'25) 用 self-evolving world model，类似 implicit memory 但仍是 single-modality。JanusVLN 创新在于**双 modality 分别建模**，类似人脑左右脑解耦。

### 6.4 与 NaVILA 对比

NaVILA (Cheng et al., RSS'25) 是 legged robot VLA 模型，基于 LLaMA-architecture，用了 13132K 数据达到 SR=54.0。JanusVLN 用 10692K（少 18%）达到 SR=60.5（高 6.5），证明 paradigm 优势而非数据优势。

参考 NaVILA：https://navila-anything.github.io/

## 7. 公式背后的 Intuition 深度

### 7.1 为什么 cross-attention with KV cache 等价于 full self-attention 的关键子集？

考虑 VGGT 原生 multi-frame attention：

$$
\text{Attn}(Q_{new}, K_{1:T}, V_{1:T}) = \text{softmax}\left(\frac{Q_{new} K_{1:T}^T}{\sqrt{d}}\right) V_{1:T}
$$

新帧 token $Q_{new}$ 与所有历史 $K, V$ 计算 attention。JanusVLN 缓存 $K_{1:t-1}, V_{1:t-1}$ 后，新帧来时只需：

$$
\text{CrossAttn}(Q_{new}, K_{cached}, V_{cached}) + \text{SelfAttn}(Q_{new})
$$

数学上等价！但工程上：
- VGGT 每次重计算所有历史 $K, V$（因为 frozen encoder 也会被多次 forward）
- JanusVLN 直接复用 cached $K, V$，避免 re-encoding

唯一区别：JanusVLN 限制 history 到 (initial + sliding) 而非全部，是 **近似**，但 ablation 显示性能损失可忽略甚至更好（attention sinks 保留全局信息）。

### 7.2 为什么 λ=0.2 最优？

设 semantic feature $S$ 的信息量远大于 spatial feature $G$（因为 Qwen2.5-VL encoder 训练数据规模是 VGGT 的几十倍）。如果用加法：

$$
F = S + \lambda \cdot \text{MLP}(G)
$$

MLP 把 $G$ 投影到 $S$ 的 channel 空间。$\lambda$ 控制 spatial 信号的"音量"。$\lambda=0.5$ 时 spatial 信号太强，淹没 semantic grounding；$\lambda=0.1$ 时太弱，3D prior 没充分利用。0.2 是 sweet spot。

类似 LoRA 中的 scaling factor $\alpha/r$，控制 adapter 信号强度。

### 7.3 Attention Sinks 在 VLN 中的特殊意义

在 NLP 中 attention sinks 是 BOS token 等无信息 token 吸收冗余 attention。在 VLN 中，**初始帧**往往是 agent 起始位置的全景观察，包含：
- 整个房间的 layout 上下文
- 起点相对目标的 global 几何参考
- 指令的 visual grounding 锚点

这些信息在整个 trajectory 中都需要反复参考，类似"导航的起点坐标系"。

## 8. 关键限制与潜在问题

### 8.1 Memory 固定大小的代价

48 帧 sliding window + 8 帧 initial = 56 帧 KV。如果 trajectory > 56 帧（实际可达数百帧），中间帧的 KV 被 evicted。Table 5 显示从 48 → 64 帧几乎无增益，说明 56 帧 enough for R2R-CE 平均轨迹长度（~10-30 步）。但**更长程任务（如 RxR 长指令）可能饱和**。

### 8.2 VGGT frozen 的限制

VGGT encoder 保持 frozen，未在 VLN 数据上微调。Table 4 显示 frozen VGGT 已带来 5.8 SR 提升，但若 end-to-end fine-tune 可能更高。frozen 选择是为节省计算和避免 catastrophic forgetting。

### 8.3 Real-world 评估规模有限

Figure 4 和 Figure 5 的 real-world 实验是 qualitative 的，仅 Unitree Go2 + Insta360 X5 几个案例。没有 quantitative real-world benchmark，sim-to-real gap 未系统评估。

### 8.4 与 LLM 推理效率技术的关联

KV cache 管理在 LLM serving 中已有成熟技术：PagedAttention (vLLM)、Sliding Window Attention (Mistral)、Ring Attention。JanusVLN 的 hybrid cache 策略与 StreamingLLM 直接对应，但**未引用更广泛的 KV cache 优化文献**——可能是 future work 方向。

参考：
- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- Mistral sliding window: https://arxiv.org/abs/2310.06825

## 9. 延伸联想与 Open Questions

### 9.1 与 DUSt3R/MASt3R 的关联

VGGT 与 DUSt3R (Naver Labs, CVPR 2024)、MASt3R 同属 feed-forward 3D reconstruction 家族。JanusVLN 的设计可推广到任何此类 3D foundation model 作为 spatial encoder。

参考 DUSt3R：https://dust3r.europe.naverlabs.com/

### 9.2 与 World Model 的关联

NavMorph 用 world model，JanusVLN 用 implicit KV memory。两者本质都是**学习 environment dynamics 的 compressed representation**。JanusVLN 的 KV cache 可视为 world model 的 implicit state——下一步 action 预测基于这个 state。

### 9.3 Scaling Laws

实验中 JanusVLN 用 7B backbone（Qwen2.5-VL 7B）。若用 72B 是否继续提升？Table 4 显示 VGGT 增益来自 prior 而非参数——暗示 scaling LLM 仍有效，3D prior 是 orthogonal 改进维度。

### 9.4 与 Embodied AI Foundation Models 的位置

JanusVLN 是 VLN-specialized。当前趋势（OpenVLA, π0, GR-2）是通用 VLA foundation model。JanusVLN 的 dual implicit memory 思想可移植到 manipulation：semantic encoder + spatial encoder + dual KV cache。

参考：
- OpenVLA: https://openvla.github.io/
- π0: https://www.physicalintelligence.company/blog/pi0
- GR-2: https://agilex-robotics.github.io/gr2/

### 9.5 与 Visual Tokens Compression 的关联

Qwen2.5-VL 的 2×2 spatial merging 是 token 压缩。JanusVLN 沿用，未探索更激进的 token pruning（如 FastV、VTW）。结合 token pruning 可进一步降低 KV cache 大小。

参考 FastV：https://arxiv.org/abs/2403.06764

### 9.6 长程导航的 memory 管理

当前 hybrid（initial + sliding）是手工设计。未来可探索：
- **Learnable memory management**：学习哪些 KV 该保留
- **Hierarchical memory**：episodic + semantic + working memory 多层级
- **Differentiable memory retrieval**：类似 differentiable neural dictionary

### 9.7 跨模态融合的更深设计

公式 (4) 用简单 addition。Table 6 显示 cross-attention 略差。但更复杂的融合（gated fusion、MoE per modality、contrastive alignment）未被探索。可能 7B 规模下数据不足，更大模型 + 更多数据下复杂融合会有不同表现。

## 10. 总结：JanusVLN 的核心贡献

1. **Paradigm shift**：从 explicit semantic memory 到 dual implicit neural memory（sem + geo）
2. **Architecture innovation**：把 3D foundation model (VGGT) 作为 spatial encoder 集成进 MLLM
3. **Efficiency breakthrough**：固定大小 KV cache，推理时间 69-90% 降低
4. **Cognitive science grounding**：左右脑分工的 dual encoder 设计

**SOTA 表现**（R2R-CE Val-Unseen）：
- SR 60.5（前 SOTA StreamVLN 56.9，+3.6）
- SPL 56.8（前 SOTA StreamVLN 51.9，+4.9）
- 仅用 10692K 训练数据（StreamVLN 用 26330K，2.5× 节省）

**核心 insight**：navigation 需要的不是更多数据，而是更结构化的 spatial-semantic 解耦表征。3D prior from RGB（通过 VGGT）+ 双 implicit KV cache 是这个 insight 的具体实现。

## 11. 我会进一步深挖的方向

如果想 build 更深 intuition，建议关注：

1. **复现 VGGT 在 VLN 中的 KV cache 实现细节**：VGGT 的 fusion decoder 中具体哪些 layer 的 KV 被 cache？是 attention output 还是 FFN 前的 hidden state？
2. **Attention pattern 可视化**：spatial encoder 和 semantic encoder 的 attention 在不同帧上如何分布？是否真的 spatial 偏 global，semantic 偏 local？
3. **Failure case analysis**：JanusVLN 在什么类型的 instruction 上失败？空间关系密集的？长程依赖的？还是 dynamic scene 的？
4. **Ablation on initial window size**：Table 5 显示去掉 initial KV 降 1.8 SR，但 initial window 大小（8 帧）是否最优？未消融。
5. **Cross-dataset generalization**：在 RxR-CE 上 SR 提升 3.3-30.7，但 RxR 多语言指令（英、印、泰）是否对 spatial encoder 公平？多语言 instruction 是否影响 semantic encoder 表征质量？

## 参考资源链接汇总

- JanusVLN 项目主页：https://miv-xjtu.github.io/JanusVLN.github.io/
- VGGT：https://vggt.github.io/
- Qwen2.5-VL：https://arxiv.org/abs/2502.13923
- StreamingLLM (Attention Sinks)：https://arxiv.org/abs/2309.17453
- VLN-CE benchmark：https://jacobkrantz.github.io/vln-ce/
- Habitat simulator：https://aihabitat.org/
- NaVILA：https://navila-anything.github.io/
- StreamVLN：https://arxiv.org/abs/2507.05240
- DUSt3R：https://dust3r.europe.naverlabs.com/
- Matterport3D：https://niessner.github.io/Matterport/
- DAgger 原始论文：https://arxiv.org/abs/1011.0686
- vLLM PagedAttention：https://arxiv.org/abs/2309.06180
- Mistral sliding window：https://arxiv.org/abs/2310.06825
- R2R benchmark：https://embodiedqa.org/
- SpatialVLM：https://spatial-vlm.github.io/
