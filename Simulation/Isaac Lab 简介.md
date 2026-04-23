## Isaac Lab 简介

**Isaac Lab** 是 NVIDIA 基于 Isaac Sim 构建的 **robot learning 框架**，主要用于在 GPU 加速的物理仿真环境中进行机器人策略训练（如 reinforcement learning）。其核心架构围绕以下几个关键点：

1. **多物理引擎支持**（3.0 Beta 起）：通过抽象接口（如 `Articulation`, `ContactSensor`）解耦底层引擎，支持 **PhysX**、**Newton** 及用户自定义物理引擎，所有默认环境可跨引擎运行，且保持与 2.X 的向后兼容。

2. **高性能底层实现**：低层代码从 PyTorch 迁移到 **NVIDIA Warp**，以利用 CUDA graph 和更高效的 GPU kernel；数据类（如 `ArticulationData`）返回 Warp 数组，需通过 `wp.to_torch()` 转换，setter 同时支持 warp 数组和 torch tensor。

3. **Quaternion 惯例统一**：默认四元数惯例从 `wxyz` 改为 `xyzw`，与 PhysX、Newton、Warp 保持一致，减少不必要的格式转换开销。

4. **实验性扩展机制**：通过 `isaaclab_experimental` 和 `isaaclab_task_experimental` 快速引入新特性（如 direct RL 任务的 CUDA graphing 支持），显著降低框架开销、加速训练。

简单来说，Isaac Lab = **Isaac Sim + 可插拔物理引擎 + Warp 加速 + RL 训练框架**，是 NVIDIA 面向机器人学习的端到端仿真与训练平台。