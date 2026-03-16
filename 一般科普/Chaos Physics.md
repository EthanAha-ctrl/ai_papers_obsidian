# Chaos Physics 深度解析

这篇文章是 **Unreal Engine** 官方关于 **Chaos Physics** 系统的概览文档。Chaos Physics 是 Epic Games 开发的轻量级物理模拟解决方案，用于替代 NVIDIA PhysX，并在 **Fortnite** 中得到实际应用。

---

## 1. Chaos Physics 核心定位

### 1.1 为什么需要 Chaos Physics？

| 对比维度 | PhysX (NVIDIA) | Chaos Physics (Epic) |
|---------|---------------|---------------------|
| **所有权** | 第三方闭源 | Epic 自主可控 |
| **集成度** | 需要外部 SDK | 引擎原生集成 |
| **定制性** | 受限于 NVIDIA API | 完全开源可定制 |
| **跨平台** | 依赖特定驱动 | 纯软件实现，跨平台一致 |
| **未来维护** | 已停止更新 | UE5 主要物理方案 |

**核心动机**：Epic 需要一个能够完全掌控、深度优化的物理引擎，避免第三方依赖带来的技术债务和授权风险。

### 1.2 五大核心模块架构

```
┌─────────────────────────────────────────────────────────┐
│                   Chaos Physics Engine                   │
├─────────────┬─────────────┬──────────┬─────────┬────────┤
│    RBAN     │  Destruction│   Cloth  │ Ragdoll │Vehicle │
│ (Rigid Body │             │          │         │        │
│  Animation) │             │          │         │        │
└─────────────┴─────────────┴──────────┴─────────┴────────┘
```

---

## 2. RBAN (Rigid Body Animation Nodes) 深度解析

### 2.1 什么是 RBAN？

**RBAN** 是一个在 Animation Blueprint 中运行的刚体动力学系统，用于创建程序化的二次运动。

**典型应用场景**：
- 角色身上的挂件、饰品物理模拟
- 头发、尾巴等附属物的动态效果
- 机械关节的物理驱动
- 布娃娃系统 的局部应用

### 2.2 RBAN 核心技术参数

#### 约束系统

**位置约束方程**：

$$C(x_1, x_2) = |x_1 - x_2| - L_0 = 0$$

其中：
- $x_1, x_2$ = 两个刚体的世界坐标位置
- $L_0$ = 约束的静止长度

**椭圆锥约束** — Chaos 独有改进：

$$\left(\frac{u_x}{a}\right)^2 + \left(\frac{u_y}{b}\right)^2 + \left(\frac{u_z}{c}\right)^2 \leq 1$$

其中：
- $u = (u_x, u_y, u_z)$ = 相对位置向量
- $a, b, c$ = 椭圆锥的三个轴半径

这比 PhysX 的圆形锥约束更灵活，可以模拟非对称的活动范围。

#### 稳定性改进

Chaos RBAN 在 **低迭代次数** 下保持更好的稳定性，这通过以下技术实现：

1. **Position-Based Dynamics (PBD)** 变体
2. **约束投影**
3. **改进的积分器**

**Verlet 积分**（常用于 PBD）：

$$x_{n+1} = 2x_n - x_{n-1} + a_n \Delta t^2$$

其中：
- $x_n$ = 当前位置
- $x_{n-1}$ = 上一帧位置
- $a_n$ = 加速度
- $\Delta t$ = 时间步长

### 2.3 RBAN World Object Support

这是一个关键特性，允许将主世界中的物体复制到 RBAN 模拟中：

```
┌─────────────────────────────────────────────────────────┐
│                  Game World Simulation                   │
│  ┌─────┐  ┌─────┐  ┌─────┐                              │
│  │Obj A│  │Obj B│  │Obj C│  ← 主世界物体                 │
│  └──┬──┘  └──┬──┘  └─────┘                              │
└─────┼────────┼───────────────────────────────────────────┘
      │        │
      ▼        ▼    复制/同步
┌─────────────────────────────────────────────────────────┐
│                   RBAN Simulation                        │
│  ┌─────┐  ┌─────┐                                       │
│  │A'   │  │B'   │  ← RBAN 模拟中的副本                  │
│  │static│  │kinematic│                                  │
│  └─────┘  └─────┘                                       │
│              ┌───────┐                                  │
│              │Ragdoll│ ← 角色 RBAN 模拟                 │
│              └───────┘                                  │
└─────────────────────────────────────────────────────────┘
```

**静态 vs 运动学**：
- **Static Objects**: 不可移动的障碍物
- **Kinematic Objects**: 由动画/脚本驱动的物体，会影响其他物理体但自身不受力

### 2.4 RBAN Simulation Space — 核心创新

#### 问题背景

在组件空间 或骨骼空间 模拟中，RBAN 不受角色移动影响，这避免了不真实运动导致的伪影，但同时也让模拟完全无法响应角色运动。

#### 解决方案：混合空间模拟

**核心参数 MasterAlpha**：

$$\alpha_{master} \in [0, 1]$$

- $\alpha = 0$: 完全局部空间模拟（不受角色移动影响）
- $\alpha = 1$: 完全世界空间模拟（完全响应角色移动）

**力的传递方程**：

$$F_{transferred} = \alpha_{master} \cdot F_{world}$$

**支持的非惯性力**：

| 力类型 | 公式 | 物理意义 |
|--------|------|----------|
| **Coriolis Force** | $\vec{F}_c = -2m(\vec{\omega} \times \vec{v})$ | 旋转参考系中的横向力 |
| **Centrifugal Force** | $\vec{F}_{cf} = -m\vec{\omega} \times (\vec{\omega} \times \vec{r})$ | 旋转产生的离心效应 |
| **Euler Force** | $\vec{F}_e = -m\frac{d\vec{\omega}}{dt} \times \vec{r}$ | 角加速度产生的力 |

其中：
- $m$ = 质量
- $\vec{\omega}$ = 角速度向量
- $\vec{v}$ = 相对速度
- $\vec{r}$ = 位置向量

---

## 3. Chaos Cloth 详细解析

### 3.1 布料模拟基础

**粒子-弹簧模型**：

布料通常被建模为质点网格，通过弹簧连接：

$$F_{spring} = -k(|x_{ij}| - L_0)\hat{x}_{ij} - c\dot{x}_{ij}$$

其中：
- $k$ = 弹簧刚度
- $L_0$ = 自然长度
- $c$ = 阻尼系数
- $x_{ij}$ = 粒子 i 到 j 的向量

### 3.2 Chaos Cloth 改进点

#### 突破 32 球体碰撞限制

PhysX 布料碰撞最多支持 32 个球体，这在复杂角色上是个严重限制。

**Chaos 解决方案**：
- 使用 **BVH (Bounding Volume Hierarchy)** 加速结构
- 支持 **任意数量** 的碰撞体
- 使用 **SDF (Signed Distance Field)** 进行精确碰撞检测

#### Wrap Deformer 改进

**包裹变形器** 用于将布料贴合到角色表面：

```
Cloth Particle → Find Nearest Surface Point → Project onto Surface
```

**质量改进**：
- 更好的法线插值
- 权重衰减优化
- 减少穿插

#### LOD (Level of Detail) 支持

```
┌─────────────────────────────────────────────────┐
│              Cloth LOD System                    │
├──────────┬──────────┬──────────┬────────────────┤
│  LOD 0   │  LOD 1   │  LOD 2   │  LOD 3...      │
│ High Res │ Medium   │ Low Res  │  Very Low      │
│ 2048 pt  │ 512 pt   │ 128 pt   │  32 pt         │
└──────────┴──────────┴──────────┴────────────────┘
          ▲
          │ 距离/性能自动切换
    Camera Distance
```

---

## 4. Chaos Ragdoll

### 4.1 核心功能

**Ragdoll** 是将角色动画与物理模拟混合的系统：

```
┌─────────────────────────────────────────────────────────┐
│                   Motion Blending                        │
│                                                          │
│   Animation Keyframes  ─────┐                           │
│                             │                           │
│                             ▼                           │
│                    ┌───────────────┐                    │
│                    │   Blend Node  │                    │
│                    │   w ∈ [0,1]   │                    │
│                    └───────┬───────┘                    │
│                            │                            │
│   Physics Simulation  ─────┘                           │
│                                                          │
│   Output = w × Animation + (1-w) × Physics              │
└─────────────────────────────────────────────────────────┘
```

### 4.2 与 PhysX 的 Parity 特性

| 特性 | 说明 |
|------|------|
| **Joint Collision** | 关节间的碰撞检测 |
| **Collision Manager** | 统一的碰撞过滤和管理 |
| **Root Animation** | 根骨骼动画支持 |
| **Sequencer PIE** | 在编辑器中播放 的 ragdoll |
| **Kinematic Character Interaction** | 运动学角色与 ragdoll 的交互 |

---

## 5. Chaos Vehicles (Experimental)

### 5.1 架构对比

```
PhysX Architecture:           Chaos Architecture:
┌──────────────┐              ┌──────────────┐
│VehicleWheel  │              │ChaosVehicle  │
│              │              │    Wheel     │
└──────────────┘              └──────────────┘
┌──────────────┐              ┌──────────────┐
│VehicleMovement│             │ChaosWheeled  │
│              │              │VehicleComp   │
└──────────────┘              └──────────────┘
┌──────────────┐              ┌──────────────┐
│VehicleAnim   │              │VehicleAnim   │
│Instance      │              │Instance      │
└──────────────┘              └──────────────┘
┌──────────────┐              ┌──────────────┐
│WheelHandler  │              │WheelController│
└──────────────┘              └──────────────┘
```

### 5.2 核心改进

#### 突破 4 轮限制

PhysX 车辆系统被设计为最多 4 个轮子。Chaos 允许 **任意数量** 的轮子，这对于：
- 多轴卡车
- 六轮装甲车
- 摩托车（2轮 + 边车）
- 坦克（履带模拟）

#### Wheel Blueprint 系统

**WheelController 节点** 自动处理：
- 转向角度
- 车轮旋转
- 悬挂压缩

### 5.3 车辆物理方程

#### 悬挂系统

**弹簧-阻尼模型**：

$$F_{susp} = -k_s \cdot \Delta x - k_d \cdot \dot{x}$$

其中：
- $k_s$ = 弹簧刚度
- $k_d$ = 阻尼系数
- $\Delta x$ = 压缩量

#### 轮胎力模型

**Pacejka Magic Formula** (常用轮胎模型)：

$$F_y = D \sin(C \arctan(B\alpha - E(B\alpha - \arctan(B\alpha))))$$

其中：
- $F_y$ = 侧向力
- $\alpha$ = 侧偏角
- $B, C, D, E$ = 拟合参数

---

## 6. Chaos Destruction

文档中提到 **APEX 与 Chaos Destruction 不兼容**。

### 6.1 Geometry Collection 系统

Chaos Destruction 使用 **Geometry Collection** 替代 APEX Destruction：

```
┌─────────────────────────────────────────────────────────┐
│              Geometry Collection Pipeline               │
├─────────────────────────────────────────────────────────┤
│  1. Fracture Tool                                       │
│     └─→ Voronoi Fracture                                │
│     └─→ Uniform Fracture                                │
│     └─→ Cluster Fracture                                │
│                                                          │
│  2. Cluster Hierarchy                                    │
│     └─→ Level 0: Root cluster                           │
│     └─→ Level 1: Sub-clusters                           │
│     └─→ Level N: Individual pieces                      │
│                                                          │
│  3. Simulation                                           │
│     └─→ Rigid body dynamics                              │
│     └─→ Collision detection                              │
│     └─→ Constraint solving                               │
└─────────────────────────────────────────────────────────┘
```

### 6.2 层级聚类

**聚类方程**：

$$C_{cluster} = \{B_1, B_2, ..., B_n\}$$

其中 $B_i$ 是子刚体。当损坏阈值被触发时，聚类分裂为子聚类或独立刚体。

---

## 7. 启用 Chaos Physics 的技术细节

### 7.1 编译开关

```csharp
// UE4Editor.Target.cs
bCompileChaos = true;  // 编译 Chaos 模块
bUseChaos = true;      // 运行时使用 Chaos
```

### 7.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────┐
│                    Engine Core                          │
├─────────────────────────────────────────────────────────┤
│                    Physics Core                         │
│         ┌───────────────────┬──────────────────┐       │
│         │    PhysX Module   │  Chaos Module    │       │
│         │    (Deprecated)   │  (Active)        │       │
│         └───────────────────┴──────────────────┘       │
├─────────────────────────────────────────────────────────┤
│                    Plugin Layer                         │
│  ┌─────────────┬─────────────┬─────────────┬──────────┐│
│  │ChaosCloth   │ChaosVehicle │ChaosDestruc │ChaosCache││
│  │  Plugin     │  Plugin     │  Plugin     │ Plugin   ││
│  └─────────────┴─────────────┴─────────────┴──────────┘│
└─────────────────────────────────────────────────────────┘
```

---

## 8. 故障排除与调试

### 8.1 Chaos Vehicles 调试命令

所有命令以 `p.vehicle` 开头：

| 命令 | 功能 |
|------|------|
| `p.vehicle.DebugDrawing 1` | 启用调试绘制 |
| `p.vehicle.ShowCOM 1` | 显示质心 |
| `p.vehicle.ShowSuspension 1` | 显示悬挂 |
| `p.vehicle.ShowWheels 1` | 显示轮子信息 |

### 8.2 Blueprint Reparenting 技巧

文档提到一个有用的技巧：通过 **Reparent Blueprint** 将 PhysX 车辆动画蓝图转换为 Chaos 版本：

```
1. 打开旧的 PhysX Vehicle Anim Blueprint
2. File → Reparent Blueprint
3. 选择 VehicleAnimationInstance
4. 删除 WheelHandler 节点
5. 添加 WheelController 节点
```

---

## 9. 关键技术洞察

### 9.1 Position-Based Dynamics (PBD) vs Force-Based Dynamics

Chaos 大量采用 **PBD** 方法，相比传统的基于力的模拟：

| 特性 | Force-Based | Position-Based |
|------|-------------|----------------|
| **稳定性** | 需要小时间步 | 大时间步稳定 |
| **约束处理** | 需要约束力 | 直接投影到约束流形 |
| **能量守恒** | 精确 | 近似 |
| **实时性** | 较差 | 优秀 |

### 9.2 为什么 Chaos 更适合游戏？

1. **确定性**：对于网络同步至关重要
2. **性能**：PBD 的稳定性允许更大的时间步
3. **可调试性**：Epic 完全掌控代码
4. **跨平台一致性**：不依赖特定硬件加速

---

## 10. 相关资源

- **官方文档**: [Chaos Physics Overview](https://docs.unrealengine.com/4.26/en-US/Physics/ChaosPhysics/)
- **Fortnite 技术分享**: [Physics in Fortnite](https://www.unrealengine.com/en-US/blog/fortnite-uses-chaos-physics)
- **UE5 物理路线图**: [UE5 Physics Roadmap](https://docs.unrealengine.com/5.0/en-US/physics-in-unreal-engine/)
- **Chaos 源码**: [Unreal Engine GitHub](https://github.com/EpicGames/UnrealEngine/tree/release/Engine/Source/Runtime/Experimental/Chaos)

---

这篇文章虽然定位为"Overview"，但实际上涵盖了 Chaos Physics 的完整架构和关键实现细节。对于从 PhysX 迁移的开发者，重点关注 **API 映射关系** 和 **调试命令**；对于深入研究物理模拟的开发者，**RBAN Simulation Space** 和 **PBD 稳定性改进** 是最有价值的技术亮点。