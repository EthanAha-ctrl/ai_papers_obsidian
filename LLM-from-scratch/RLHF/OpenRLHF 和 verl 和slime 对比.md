### 1. OpenRLHF：disaggregated解耦架构

OpenRLHF采用**Ray + vLLM混合分布式架构**，核心创新是将不同模型角色分离到独立的GPU资源池中：

```
[Actor Policy] ←Ray RPC→ [Reference Policy] ←Ray RPC→ [Reward Model]
       ↓                            ↓                      ↓
   GPU Pool A                  GPU Pool B             GPU Pool C
       |                            |                      |
       +------------------[Training Loop with DeepSpeed]--+
```

- `create_vllm_engines()`函数创建多个`vLLM LLMEngine`实例作为Rayactor，每个绑定特定GPU placement group
- 使用**Ray Placement Groups**管理资源分配，确保vLLM engine和训练进程的GPU隔离
- 训练阶段采用**DeepSpeed ZeRO-3**实现3D并行：
  - Tensor Parallelism (TP): 自动张量并行，通过`AutoTP`实现
  - Data Parallelism (DP): 基于ZeRO的参数/梯度/优化器状态分片
  - Pipeline Parallelism (PP): 序列并行，减少流水线气泡

根据论文`https://arxiv.org/pdf/2405.11143`，OpenRLHF在不同模型规模上实现**1.22x–1.68x**的训练加速比。

---

### 2. verl (HybridFlow)：混合编程模型

verl的最大创新是**HybridFlow**编程范式，解耦数据流（dataflow）和计算（computation）：

```
[Single-Controller] ↔ [Multi-Controller]
       ↓                    ↓
   {数据调度}      {GPU协同计算}
```

**内存优化技术**：
- **colocated actor+reference**：将actor policy和reference policy放在同一GPU，因为reference在LoRA PPO中是actor的基线模型，这样可以共享基础模型权重，减少50%内存
- **数据流与计算分离**：通过`Ray actor groups`灵活组合不同角色的GPU分配，支持三种模式：
  - 所有模型集中放置（Colocation mode）
  - 每个模型独立GPU（Disaggregated mode）  
  - 混合模式：actor/ref共享GPU，critic/RM另配GPU

**官方文档**：`https://verl.readthedocs.io/en/latest/hybrid_flow.html`详细说明了hybrid controller设计如何提供计算效率和灵活性。

**PPO实现**：
`RayPPOTrainer`运行在driver进程（通常是CPU节点），通过Ray RPC协调worker group。支持：
- LoRA集成，通过`actor_rollout_ref.model.lora.merge`参数控制是否在rollout前合并LoRA适配器
- 异步 rollout，最大化GPU利用率

---

### 3. slime：SGLang-native Megatron集成
核心特点是**SGLang原生集成**：

```
[Megatron-LM Training Backend] ↔ [SGLang Rollout Engine]
       ↓ (ZeRO-3, TP, PP)              ↓ (Continuous Batching, PagedAttention)
       +-----------[RL Controller: PPO/GRPO]------------+
```

- **Megatron-SGLang桥接**：训练使用Megatron-LM的分布式训练能力（支持TP/PP/ZeRO），rollout使用SGLang的高效推理引擎
- **SGLang优势**：原生支持interleaved text-generation, 连续批处理（continuous batching），自动KV cache管理
- **MoE支持**：专门优化MoE模型，如GLM-4 MoE架构（参考GitHub issue #806关于`glm4_moe_lite`支持）
- **CUDA Graph集成**：利用SGLang的CUDA graph capture加速rollout，但早期版本有hang的问题（issue #1484）

**应用案例**：
- GLM-4.5 和 GLM-4.6 模型的RL训练
- 支持Qwen3系列（根据Facebook帖子）
- Z.ai公司的模型也使用slime框架

---
### 易用性
- **OpenRLHF**: 文档完善（`https://openrlhf.readthedocs.io/`），但GitHub issues显示存在vLLM兼容性问题（issue #368关于断点续训）
- **verl**: 灵活的HybridFlow API学习曲线稍陡，但文档详细（`https://verl.readthedocs.io/en/latest/`）
- **slime**: 特定于THUDM生态，如果使用GLM或需要MoE支持是最佳选择

### 社区活跃度
- **OpenRLHF**: 社区较大，issues和discussions活跃，roadmap公开（issue #568）
- **verl**: 字节跳动维护，工业级应用，文档更新频繁
- **slime**: 主要服务于THUDM内部模型，但开源版本持续更新（GitHub release频繁，PR数量92个）

---

## 第一性原理分析

RLHF训练的核心瓶颈在于：

1. **来回通信开销**：actor rollout → 奖励计算 → policy gradient update → 新actor参数同步
2. **内存墙**：需要同时加载actor、reference、critic、reward模型
3. **生成速度**：rollout阶段是主要耗时部分，受限于推理引擎的吞吐

三个框架的解决思路：

- **OpenRLHF**：通过**解耦+专业化引擎**，让vLLM专攻生成，DeepSpeed专攻训练，用Ray做资源调度
- **verl**：通过**数据流抽象+内存共享**，减少不必要的数据传输和内存占用，尤其适合LoRA场景
- **slime**：通过**后端紧耦合**，让Megatron和SGLang深度集成，优化特定模型（GLM）的端到端性能

---

## 选择建议

| 场景 | 推荐框架 | 理由 |
|------|---------|------|
| 通用RLHF，多模型支持 | OpenRLHF | 生态最完善，vLLM推理快 |
| 大规模生产，内存受限 | verl | HybridFlow灵活性高，colocation节省内存 |
| GLM-4.x系列或MoE模型 | slime | 原生支持，性能优化到位 |
| 需要LoRA训练 | verl | LoRA集成最成熟 |
| 需要长上下文生成 | OpenRLHF | vLLM的PagedAttention处理长序列有优势 |

---

**关键参考链接**：
- OpenRLHF论文: `https://arxiv.org/pdf/2405.11143`
- verl HybridFlow论文: `https://arxiv.org/abs/2409.19256`
- slime文档: `https://thudm.github.io/slime/`
- OpenRLHF vs verl深度对比: `https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep`
- verl官方文档: `https://verl.readthedocs.io/en/latest/hybrid_flow.html`
- OpenRLHF GitHub: `https://github.com/openrlhf/openrlhf`
- verl GitHub: `https://github.com/volcengine/verl`
- slime GitHub: `https://github.com/THUDM/slime`