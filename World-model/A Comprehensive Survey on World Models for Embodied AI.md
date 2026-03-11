我将分段读取这篇综述文章的完整内容，然后为您提供详细的技术讲解。



我将为您提供这篇关于**Embodied AI中World Models的综合综述**的详细技术讲解。这篇文章是2025年发表的综述，对世界模型领域进行了系统性的梳理。

---

## 一、核心概念与数学基础

### 1.1 什么是World Model？

World Model是**环境动力学的内部模拟器**，它使agent能够：
- **Simulation & Planning**: 通过学习到的动力学生成合理的未来场景，让agent通过"想象"评估潜在动作
- **Temporal Evolution**: 学习编码状态如何演化，实现时间一致的rollout
- **Spatial Representation**: 以适当的保真度编码场景几何

### 1.2 POMDP数学形式化

文章将环境交互形式化为部分可观察马尔可夫决策过程（POMDP）。定义空初始动作a₀（t=0时），使动力学写法统一。在每一步t≥1时：

- agent接收观察oₜ并采取动作aₜ
- 真实状态sₜ保持不可观测

核心公式：

**动力学先验：**
```
p_θ(zₜ | zₜ₋₁, aₜ₋₁)
```
其中zₜ是t时刻的latent状态，aₜ₋₁是t-1时刻的动作

**滤波后验：**
```
q_ϕ(zₜ | zₜ₋₁, aₜ₋₁, oₜ)
```
使用一步滤波后验推断latent状态，zₜ₋₁假设总结了相关历史

**重构：**
```
p_θ(oₜ | zₜ)
```
从latent状态重构观察

**联合分布分解：**
```
p_θ(o₁:T, z₀:T | a₀:T₋₁) = p_θ(z₀) ∏ₜ₌₁ᵀ p_θ(zₜ | zₜ₋₁, aₜ₋₁) p_θ(oₜ | zₜ)
```

**变分后验：**
```
q_ϕ(z₀:T | o₁:T, a₀:T₋₁) = q_ϕ(z₀ | o₁) ∏ₜ₌₁ᵀ q_ϕ(zₜ | zₜ₋₁, aₜ₋₁, oₜ)
```

**ELBO目标函数：**
```
ℒ(θ, ϕ) = ∑ₜ₌₁ᵀ 𝔼_qϕ(zₜ)[log p_θ(oₜ | zₜ)] - D_KL(q_ϕ(z₀:T | o₁:T, a₀:T₋₁) ‖ p_θ(z₀:T | a₀:T₋₁))
```

- 重构项：log p_θ(oₜ | zₜ) 鼓励忠实的观察预测
- KL正则化：将滤波后验与动力学先验对齐

---

## 二、三轴分类框架

### 轴1：功能性耦合

| 类型 | 特点 | 代表方法 |
|------|------|----------|
| **Decision-Coupled** | 任务特定，针对特定决策任务学习动力学 | Dreamer系列、PlaNet、TransDreamer |
| **General-Purpose** | 任务无关的模拟器，专注于广泛预测 | Sora、V-JEPA 2、Genie |

### 轴2：时间建模

| 类型 | 特点 | 优势 | 劣势 |
|------|------|------|------|
| **Sequential Simulation and Inference** | 自回归方式，逐步展开未来状态 | 紧凑、样本高效 | 误差累积 |
| **Global Difference Prediction** | 并行估计整个未来状态 | 减少误差累积 | 计算成本高、交互性弱 |

### 轴3：空间表示

1. **Global Latent Vector (GLV)**: 将复杂世界状态编码为紧凑向量
2. **Token Feature Sequence (TFS)**: 将世界状态建模为token序列
3. **Spatial Latent Grid (SLG)**: 利用BEV特征或体素网格等几何先验
4. **Decomposed Rendering Representation (DRR)**: 使用3DGS或NeRF等可学习原语

---

## 三、代表性方法详解

### 3.1 Decision-Coupled + Sequential Simulation

#### RSSM (Recurrent State-Space Model)

架构图解析：
```
观察oₜ → 编码器 → 确定性记忆hₜ + 随机状态zₜ
           ↑
动作aₜ₋₁ + hₜ₋₁ → RSSM核心 → hₜ, zₜ
                          ↓
                     解码器 → oₜ重构
```

Dreamer系列的演进：
- **PlaNet** (ICML'19): 首次引入RSSM
- **Dreamer** (ICLR'20): 引入imagination-based policy优化
- **DreamerV2** (ICLR'21): 改进样本效率
- **DreamerV3** (Nature'25): 大规模预训练

#### TransDreamer (arXiv'22)

使用Transformer State-Space Model (TSSM) 替换RNN核心：
```
多头自注意力机制捕捉长期依赖
位置编码维护时间序列顺序
并行训练支持大规模数据
```

#### GLAM (AAAI'25)

基于Mamba的状态空间模型：
```
Mamba层：线性时间复杂度 + 长期建模能力
全局模块：捕捉上下文动态
局部模块：细粒度动态建模
```

### 3.2 General-Purpose + Sequential Simulation

#### V-JEPA 2 (arXiv'25)

Joint-Embedding Predictive Architecture：
```
输入视频 → 编码器 → 嵌入序列
              ↓
        掩码预测器 → 预测被掩码区域的潜在特征
              ↓
        无需像素重构或对比学习
```

关键创新：
- 在大规模互联网视频上预训练（VideoMix22M）
- 结合有限机器人交互数据进行后训练
- 转移到机器人规划

#### Sora (arXiv'24)

视频生成范式：
```
视频 → 统一时空Patch → DiT生成器
                     ↓
              长时、连贯的序列生成
```

### 3.3 空间表示方法

#### Spatial Latent Grid - OccWorld (ECCV'24)

Occupancy预测架构：
```
多相机输入 → TPVFormer → TPV特征
                        ↓
                     Transformer → 未来Occupancy预测
                        ↓
                     体素解码器 → 3D场景表示
```

#### Decomposed Rendering - 3DGS方法

ManiGaussian (ECCV'24)：
```
当前状态 + 动作 → 每点变化预测
                    ↓
              未来高斯场景生成
                    ↓
              操控任务的高保真预测
```

4DGS扩展：
```
时间维度添加到3DGS
动态高斯原语建模
支持新视角合成
```

---

## 四、数据资源

### 4.1 模拟平台

| 平台 | 年份 | 任务 | 输入 | 特点 |
|------|------|------|------|------|
| MuJoCo | 2012 | 连续控制 | Proprioception | 高效物理模拟 |
| CARLA | 2017 | 驾驶模拟 | RGB-D/Seg/LiDAR | 基于Unreal Engine |
| Habitat | 2019 | 具身导航 | RGB-D/Seg/GPS | 光照渲染、高性能 |
| Isaac Lab | 2023 | 机器人学习 | 多模态 | GPU加速、大规模RL |

### 4.2 离线数据集

| 数据集 | 规模 | 任务 | 模态 | 特点 |
|--------|------|------|------|------|
| RT-1 | 130k轨迹 | 真实机器人操作 | RGB/Text | 13台机器人，17个月 |
| OXE | 1M+轨迹 | 跨具身预训练 | RGB-D/LiDAR/Text | 21个机构，60个来源 |
| nuScenes | 1k场景 | 驾驶感知 | RGB/LiDAR/Radar | 360度传感器套件 |
| Waymo | 1.15k场景 | 驾驶感知 | RGB/LiDAR | 1200万3D和2D标注 |
| OpenDV | 2k+小时 | 驾驶视频预训练 | RGB/Text | 40+国家，244城市 |

### 4.3 真实机器人平台

| 平台 | 特点 | 应用 |
|------|------|------|
| Franka Emika | 7-DoF，1kHz扭矩控制 | 精密操作 |
| Unitree Go1 | 四足，4.7m/s，全景深度感知 | 运动与导航 |
| Unitree G1 | 43-DoF，120 N·m膝扭矩 | 人形操作 |

---

## 五、评估指标详解

### 5.1 像素生成质量

#### Fréchet Inception Distance (FID)

```
FID(x, y) = ‖μₓ - μᵧ‖₂² + Tr(Σₓ + Σᵧ - 2(ΣₓΣᵧ)^(1/2))
```

变量解释：
- μₓ, μᵧ: 真实和生成图像分布的均值向量
- Σₓ, Σᵧ: 真实和生成图像分布的协方差矩阵
- Tr(·): 矩阵迹（对角线元素之和）

**物理意义**: 比较两个高斯分布之间的Fréchet距离，惩罚保真度损失（均值偏移）和模式崩溃（协方差不匹配）

#### Fréchet Video Distance (FVD)

使用I3D网络在Kinetics-400上预训练，在运动感知特征上应用相同的Fréchet距离公式

#### Structural Similarity Index Measure (SSIM)

```
SSIM(x, y) = (2μₓμᵧ + C₁)(2Σₓᵧ + C₂) / (μₓ² + μᵧ² + C₁)(Σₓ² + Σᵧ² + C₂)
```

变量解释：
- μₓ, μᵧ: 两个图像块的均值
- Σₓ², Σᵧ²: 两个图像块的方差
- Σₓᵧ: 两个图像块的协方差
- C₁, C₂: 稳定常数（防止除零）

#### Learned Perceptual Image Patch Similarity (LPIPS)

```
LPIPS(x, y) = Σₗ (1/HₗWₗ) Σₕ,ₙ ‖wₗ ⊙ (f̂ₕ,ₙ,ₓˡ - f̂ₕ,ₙ,ᵧˡ)‖₂²
```

变量解释：
- f̂ₕ,ₙ,ₓˡ: 输入x在第l层的归一化激活
- wₗ: 通道级权重
- Hₗ, Wₗ: 特征图的高度和宽度
- ⊙: 元素级乘法

### 5.2 状态级理解

#### mean Intersection over Union (mIoU)

```
IoU_c = TP / (TP + FP + FN)
mIoU = (1/|C|) Σ_c∈C IoU_c
```

变量解释：
- TP, FP, FN: 真阳性、假阳性、假阴性
- C: 类别集合
- c: 特定类别

#### mean Average Precision (mAP)

```
AP_c,τ = ∫₀¹ P_c,τ(r) dr
mAP = (1/|C|) Σ_c∈C (1/|T|) Σ_τ∈T AP_c,τ
```

变量解释：
- P_c,τ(r): 精确率-召回率包络线
- C: 类别集合
- T: IoU阈值集合

#### Chamfer Distance (CD)

```
CD(S₁, S₂) = Σₓ∈S₁ min_ᵧ∈S₂ ‖x - y‖₂² + Σᵧ∈S₂ minₓ∈S₁ ‖x - y‖₂²
```

适用于点云、表面、占用、BEV和3D结构

### 5.3 任务性能

- **Success Rate (SR)**: 满足预定成功条件的评估片段比例
- **Sample Efficiency (SE)**: 达到目标性能所需的样本数
- **Reward**: 累积回报 Gₜ = Σₖ₌₀^∞ γᵏ rₜ₊ₖ₊₁
- **Collision Rate**: 发生碰撞的评估片段比例

---

## 六、性能比较分析

### 6.1 视频生成

在nuScenes上的性能（数值越低越好）：

| 方法 | 分辨率 | FID↓ | FVD↓ |
|------|--------|------|------|
| DrivePhysica | 256×448 | **4.0** | 38.1 |
| MiLA | 360×640 | 4.1 | **14.9** |
| GeoDrive | 480×720 | 4.1 | 61.6 |
| Vista | 576×1024 | 6.9 | 89.4 |
| GEM | 576×1024 | 10.5 | 158.5 |

**趋势分析**：
- 更高分辨率需要更大计算资源
- DrivePhysica在视觉保真度方面最优
- MiLA在时间一致性方面最优

### 6.2 4D Occupancy预测

Occ3D-nuScenes上的性能（数值越高越好）：

| 方法 | 输入 | mIoU(%)↑ | 1s IoU(%)↑ |
|------|------|----------|-----------|
| DTT-O | Occ | 85.50 | 37.69 |
| DOME-O | Occ + GT ego | 83.08 | 35.11 |
| OccWorld-O | Occ | 66.38 | 25.78 |
| OccLLaMA-O | Occ | 75.20 | 25.05 |

**关键观察**：
- 使用Occupancy输入优于仅相机输入
- 辅助监督（GT ego轨迹）进一步减轻2-3秒的性能衰减
- DTT在所有方法中表现最佳

### 6.3 控制任务

#### DMC上的Episode Return（数值越高越好）：

| 方法 | Step预算 | Reacher Easy | Cheetah Run | Avg./Total |
|------|----------|--------------|-------------|------------|
| Dreamer | 5M | 935 | 895 | 823/20 |
| HRSSM | 500k | 910 | - | 938/3 |
| TransDreamer | 2M | - | 865 | 893/4 |

#### RLBench上的Success Rate（%）：

| 方法 | Stack Blocks | Close Jar | Open Drawer | Avg./Total |
|------|--------------|-----------|-------------|------------|
| VidMan | 48 | 88 | 94 | **67/18** |
| TesserAct | - | 44 | 80 | 63/10 |
| ManiGaussian | 12 | 28 | 76 | 45/10 |

**发现**：
- IDM（Inverse Dynamics Model）是promising的架构方向
- 多模态输入和强骨干网络（3DGS、DiT）越来越普遍

---

## 七、开放挑战与未来方向

### 7.1 数据与评估

**挑战**：
- 缺乏统一的大规模数据集
- FID/FVD等指标强调像素保真度而忽略物理一致性、动力学和因果关系
- 跨领域标准缺失

**未来方向**：
- 构建统一的多模态、跨领域数据集
- 发展超越感知真实性的评估框架，评估物理一致性、因果推理和长期动力学

### 7.2 计算效率

**挑战**：
- Transformer和Diffusion网络高推理成本与实时控制需求冲突
- RNN和全局潜在向量仍广泛使用，但长期依赖建模能力有限

**未来方向**：
- 量化、剪枝和稀疏计算
- 探索SSM（如Mamba）等新型时间方法

### 7.3 建模策略

**核心挑战**：平衡循环模拟和全局预测
- 自回归设计：紧凑、样本高效，但误差累积
- 全局预测：改善多步连贯性，但计算繁重、闭环交互性弱

**空间权衡**：
- 潜在向量 vs Token序列 vs 空间网格：效率 vs 表达力
- NeRF/3DGS：高保真但动态场景扩展性差

**未来方向**：
- SSM（如Mamba）：线性时间扩展性 + 长期推理
- 掩码方法（如JEPA）：改善表示学习 + 效率
- 混合方法：结合自回归和全局预测的优势
- 显式记忆或分层规划增强长期预测稳定性
- CoT启发的任务分解改善时间一致性

---

## 八、架构图与技术对比

### 8.1 RSSM vs TSSM vs Mamba

```
RSSM (Dreamer系列):
输入 → 编码器 → 确定性记忆 → 随机状态 → 解码器
     ↑       ↑         ↑          ↑         ↑
   观察    RNN隐藏    噪声采样    重构损失  状态预测

TSSM (TransDreamer):
输入 → 编码器 → Token序列 → 多头自注意力 → 解码器
                         ↑
                    长期依赖建模

Mamba (GLAM):
输入 → 编码器 → 状态空间层 → 线性注意力 → 解码器
                  ↑
            线性时间复杂度
```

### 8.2 Diffusion-based World Models

```
视频扩散 (Sora, Vista):
噪声 → DiT → 清晰视频
      ↑
  时空Patch

空间网格扩散 (DOME, GEM):
BEV特征 → DiT → 未来网格
         ↑
     空间结构先验

3D扩散 (DriveDreamer4D):
高斯原语 → 4DGS → 动态场景
         ↑
     可微渲染
```

---

## 九、关键技术趋势总结

### 9.1 架构演进

```
早期: RNN-based (RSSM)
  ↓
中期: Transformer-based (TSSM, IRIS)
  ↓
当前: Diffusion-based (Sora, Vista) + SSM (Mamba, GLAM)
  ↓
未来: 混合架构（RNN + Transformer + Diffusion）
```

### 9.2 表示发展

```
潜在向量 (GLV) → Token序列 (TFS) → 空间网格 (SLG) → 可微渲染 (DRR)
```

### 9.3 功能扩展

```
决策耦合 (Dreamer) → 通用目的 (Sora, V-JEPA) → 多模态 (WorldVLA)
```

---

## 十、参考文献与资源链接

### 核心方法
- [Dreamer](https://arxiv.org/abs/2010.02193) - Imagination-based policy learning
- [DreamerV2](https://arxiv.org/abs/2010.02193) - Mastering Atari with visual models
- [DreamerV3](https://arxiv.org/abs/2301.04104) - General RL with large-scale pretraining
- [Sora](https://arxiv.org/abs/2403.03208) - Large-scale video generation
- [V-JEPA 2](https://arxiv.org/abs/2501.05214) - Scalable self-supervised video learning
- [Genie](https://arxiv.org/abs/2312.13729) - Generative interactive environments

### 数据集
- [Open X-Embodiment (OXE)](https://arxiv.org/abs/2310.08864) - Cross-embodiment pretraining
- [nuScenes](https://arxiv.org/abs/1903.11027) - Large-scale autonomous driving
- [Waymo](https://arxiv.org/abs/1912.04838) - Autonomous driving benchmark
- [OpenDV](https://arxiv.org/abs/2311.17903) - Driving video pretraining
- [VideoMix22M](https://arxiv.org/abs/2501.05214) - Large-scale video pretraining

### GitHub资源
- [AwesomeWorldModels](https://github.com/Li-Zn-H/AwesomeWorldModels) - Curated bibliography from this survey

---

## 十一、我的见解与联想

### 为什么需要World Models？

类比人类认知：
- 我们不直接与环境交互，而是先在"心理模型"中模拟
- 这减少试错成本，提高决策效率
- 支持"反事实推理"（what-if思维）

### 技术哲学角度

世界模型本质上是**环境动力学的可微分模拟器**：
- 传统RL: 黑盒环境交互
- Model-based RL: 可微分环境模型
- World Models: 通用环境模拟器

### 跨领域联想

- **神经科学**: 前额叶皮层的预测编码理论
- **控制理论**: 模型预测控制（MPC）
- **因果推理**: 反事实推理和干预
- **元学习**: 学习如何学习环境动态

### 未来可能的发展方向

1. **神经符号融合**: 将世界模型与符号推理结合
2. **多时间尺度建模**: 同时捕捉快速动力学和慢速变化
3. **跨模态一致性**: 视觉、语言、触觉的联合建模
4. **个性化适应**: 从少量交互中快速适应新环境
5. **可解释性**: 理解世界模型学到的物理规律

---

希望这份详细的技术讲解帮助您建立了对Embodied AI中World Models的直觉理解！这篇文章的核心贡献是提供了一个统一的三轴分类框架，为这一快速发展领域的系统性理解提供了基础。