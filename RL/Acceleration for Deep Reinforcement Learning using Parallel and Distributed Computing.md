---
source_pdf: Acceleration for Deep Reinforcement Learning using Parallel and Distributed
  Computing.pdf
paper_sha256: 1d42e6fd232e87c8354220ebdeda248483b49bbef6f81a9cf0e0728256b6efdf
processed_at: '2026-07-17T23:55:57-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Acceleration for Deep Reinforcement Learning using Parallel and Distributed Computing: A Survey 深度解读

这篇 paper 是 National University of Defense Technology 的 Zhihong Liu 等人于 2023 年发表在 ACM Computing Surveys 上的综述，系统性地梳理了 parallel and distributed computing 在 DRL 训练加速方面的方法论。下面我从直觉出发，逐层拆解。

---

## 1. 核心动机：为什么 DRL 需要 distributed computing

DRL 与 supervised learning 的本质差异在于 **没有 labeled dataset**，agent 必须通过 environment interaction 生成 experience data $\left(s_t, a_t, s_{t+1}, r_{t+1}\right)$。这带来两个 scaling bottleneck：

1. **Sample generation bottleneck**：DQN 在 Atari 上需要 50 million frames，约 38 天；PointGoal navigation 在 3D 环境中需要 2.5 billion frames。
2. **Computation bottleneck**：neural network 的 inference 与 backpropagation 本身就是 compute-intensive。

这两个 bottleneck 在 distributed setting 下还会相互纠缠，形成 heterogeneous workloads。paper 在 Section 2.3 总结了五大挑战：

- Component orchestration（Actor / Learner / Parameter Server / Replay Memory 如何组织）
- Simulation parallelism
- Heterogeneous workload 的 computing parallelism
- Gradient staleness 与 synchronization
- Beyond backpropagation 的优化路径

---

## 2. System Architectures：Centralized vs Decentralized

### 2.1 四大抽象组件

paper 把所有 distributed DRL framework 抽象为四个角色：

- **Actor**：与环境交互，执行 action，收集 $\left(s, a, r, s'\right)$
- **Learner**：从 experience 中 sample batch，在 GPU 上计算 gradient 并更新 model
- **Parameter Server**：维护 global model 的 latest parameters
- **Replay Memory**：存储 experience（仅 off-policy 场景）

### 2.2 Centralized Architecture（Fig. 3a）

Star topology，中心节点维护 global model。代表工作：

**Gorila** (Nair et al., 2015, [arXiv:1507.04296](https://arxiv.org/abs/1507.04296))：第一个大规模 distributed DRL framework。结构：
```
Actors (many) → Replay Memory ← Learners (many) → Parameter Server → Actors
```
在 Atari 上实现 10× speedup。但 Parameter Server 成为 single point of failure 与 communication bottleneck。

**Ape-X** (Horgan et al., 2018, [arXiv:1803.00933](https://arxiv.org/abs/1803.00933))：引入 **Prioritized Experience Replay (PER)**，priority 基于绝对 TD error $|\delta|$：
$$
\delta_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-) - Q(s_t, a_t; \theta)
$$
其中 $\theta^-$ 是 target network parameters，$\gamma$ 是 discount factor。360 CPU cores + 1 P100 GPU 达到 50K FPS。

**R2D2** (Kapturowski et al., 2018, [arXiv:1802.01561](https://arxiv.org/abs/1802.01561))：在 Ape-X 基础上引入 LSTM 处理 partial observability，256 actors + 1 GPU。

**A3C** (Mnih et al., 2016, [arXiv:1602.01783](https://arxiv.org/abs/1602.01783))：on-policy 的 centralized 变体。Actor 与 Learner 封装在同一 thread（actor-learner thread），多个 thread 异步向 global network push/pull gradients。16 CPU cores 超越 DQN on K40 GPU，时间减半。

### 2.3 Decentralized Architecture（Fig. 3b）

无中心节点，多个 Learner 通过 **All-Reduce** 同步 gradients，fully connected topology。

**IMPALA** (Espeholt et al., 2018, [arXiv:1802.01594](https://arxiv.org/abs/1802.01594))：CPU actors + GPU learners，引入 **V-trace** off-policy correction：
$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{i=s}^{t} c_i \right) \delta_t V
$$
其中 $\delta_t V = \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))$，$c_i = \min(\bar{c}, \pi(x_i)/\mu(x_i))$ 是 truncated IS coefficient，$\rho_t = \min(\bar{\rho}, \pi(x_t)/\mu(x_t))$。$\bar{c}, \bar{\rho}$ 控制 bias-variance tradeoff。500 CPU + 8 P100 GPU 达到 250K FPS。

**DD-PPO** (Wijmans et al., 2020, [arXiv:1911.00357](https://arxiv.org/abs/1911.00357))：256 V100 GPUs，196× speedup over single GPU。引入 **straggler preemption**——当一定比例 worker 完成经验采集后，preempt 慢 worker。

**rlpyt** (Stooke & Abbeel, 2019, [arXiv:1909.01502](https://arxiv.org/abs/1909.01502))：8 P100 GPU，6× speedup over single GPU。

### 2.4 Gossip-based 架构

Assran et al. (NeurIPS 2019, [arXiv:1906.04585](https://arxiv.org/abs/1906.04585)) 提出 **Gossip-based Actor-Learner**，peer-to-peer topology，每个 worker 只与邻居通信。理论上保证 workers 的 model parameters 偏差 bounded by $\epsilon$-neighborhood：
$$
\|W_i - W_j\| \leq \epsilon, \quad \forall i, j
$$
相比 star topology 减少单点瓶颈，相比 fully connected 减少通信开销。

### 2.5 Fine-grained Worker 分工

近期工作进一步细化角色：
- **SampleFactory** (Petrenko et al., 2020, [arXiv:2002.12444](https://arxiv.org/abs/2002.12444))：Rollout worker + Policy worker + Learner
- **SEED_RL** (Espeholt et al., 2020, [arXiv:1910.06591](https://arxiv.org/abs/1910.06591))：Actor 只做 environment interaction，inference 集中在 Learner 侧的 TPU
- **SRL** (Mei et al., ICLR 2024)：15K CPU + 32 A100 GPU，5× speedup over OpenAI Rapid

---

## 3. Simulation Parallelism：CPU Clusters vs GPU/TPU Batch

### 3.1 CPU-based Distributed Simulation

传统方式，每个 environment instance 跑在一个 process 或 thread 上。Ape-X 用 360 CPU cores 达到 50K FPS。

**核心问题**（Fig. 5a）：CPU-GPU 之间的 data ping-pong。每一步：
1. CPU 执行 simulation step → 产生 $\left(s, r\right)$
2. Copy $s$ 到 GPU memory → inference → 产生 $a$
3. Copy $a$ 回 CPU → 执行下一步

这个 context-switching overhead 在大规模场景下成为 dominant bottleneck。

### 3.2 GPU Batch Simulation（Zero-copy）

**Liang et al.** (2018, [arXiv:1810.05762](https://arxiv.org/abs/1810.05762))：GPU-accelerated RL simulator，单 GPU + CPU 达到 60K FPS，1000 humanoid robots 并行，20 分钟训练完成（相比之前 1000× CPU cores）。

**CuLE** (Dalton & Frosio, NeurIPS 2020, [arXiv:1907.08424](https://arxiv.org/abs/1907.08424))：CUDA Learning Environment，Atari emulation 完全在 GPU memory 内运行，4096 parallel environments，155K FPS。

**Isaac Gym** (Makoviychuk et al., 2021, [arXiv:2108.10470](https://arxiv.org/abs/2108.10470))：NVIDIA 的 robotics simulation platform。关键设计是 **Tensor API**——直接访问 GPU buffer 中的 simulation results，无需 CPU-GPU data transfer（Fig. 5b）。数千 environments on 单 GPU，300× speedup。

**Brax** (Freeman et al., 2021, [arXiv:2106.13281](https://arxiv.org/abs/2106.13281))：Google 开源，基于 TPU 的 rigid body simulation。TPUv3 8×8 上达到 hundreds of millions of steps/sec for MuJoCo-ant。

### 3.3 Scene Sharing 优化

由于 texture 和 geometry objects 体积大，naively 为每个 environment 加载一份不可承受。Shacklett et al. (ICLR 2021) 维护 $K \ll N$ 个 unique scenes，其中 $N$ 是 parallel environments 数量，多个 environment 共享同一 scene。

### 3.4 Sim-to-Real Transfer

关键技巧：
1. **Sensor noise injection**：在 observation 上加 realistic noise
2. **Domain randomization**：随机化 friction coefficient、object dynamics
3. **Disturbance training**：训练时随机 push robot、给 reward 加 noise

代表工作：OpenAI Dactyl ([arXiv:1808.00177](https://arxiv.org/abs/1808.00177))、Loquercio et al. (Science Robotics 2021, [DOI:10.1126/scirobotics.abg5810](https://doi.org/10.1126/scirobotics.abg5810))。

---

## 4. Computing Parallelism：三种范式

### 4.1 Cluster Computing

多机协同，high-speed interconnect。早期 MapReduce 用于 tabular RL 的 MDP 求解。DistBelief ([Dean et al., NeurIPS 2012](https://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks)) 开创 distributed DL training。Gorila 是 DRL 领域的启蒙工作。

### 4.2 Single Machine Parallelism

Multiprocessing（多 CPU）+ Multithreading（多核）。

**A3C**：16 CPU cores，2× speedup over K40 GPU implementation。
**SampleFactory**：36 CPU cores + 1 RTX 2080Ti，130K FPS，4× speedup over SEED_RL。
**GA3C** (Babaeizadeh et al., 2017, [arXiv:1611.06256](https://arxiv.org/abs/1611.06256))：trainer thread 收集 batch 后 submit 到 GPU，16 CPU + 1 Titan X，45× speedup over A3C。

### 4.3 Specialized Hardware

**GPU**：IMPALA 用 NVIDIA DGX-1（8× Tesla V100）。多 GPU 方案中最多 32 GPU（Liang et al. 2018）。

**FPGA**：
- **NNQL** (Su et al., 2017)：Arria 10 AX066 FPGA，346× speedup over GTX 760 GPU
- **FA3C** (Cho et al., ASPLOS 2019)：Xilinx VCU1525，27.9% better IPS than Tesla P100
- **PPO_FPGA** (Meng et al., FCCM 2020)：Xilinx Alveo U200，27.5× speedup over Titan Xp
- **On-chip replay** (Meng et al., 2022)：4.3× higher IPS over GTX 3090

**TPU**：
- **AlphaZero**：5000 TPU v1 + 64 TPU v2 cores，24 小时训练击败世界冠军程序
- **AlphaStar**：3072 TPU v3 + 50400 CPU cores，44 天训练达到 99.8% human player 水平
- **OpenAI Five**：1536 GPU + 172800 CPU cores，10 个月训练击败 Dota 2 世界冠军 OG
- **SEED_RL**：520 CPU + 8 TPU v3，11× faster than IMPALA on P100 GPU

### 4.4 Task Scheduling 策略

1. **Load balancing**：Ray 用 fine-grained + coarse-grained 两级；rlpyt 形成 alternating groups
2. **Resource dynamic adjusting**：MINIONSRL 动态调整 Actor 数量
3. **Computation-communication overlap**：PEARL 设计 Learner module overlap 通信与计算
4. **Preemptive scheduling**：DD-PPO 的 straggler preemption

---

## 5. Distributed Synchronization Mechanisms

### 5.1 Asynchronous Off-policy Training（Fig. 6a）

每个 worker 独立运行，不等待其他 worker。Actor 的 behavior policy $\mu$ 与 Learner 的 target policy $\pi$ 逐渐 diverge。

**Policy-lag 问题**：training data 由过时的 behavior policy 产生，distribution shift 导致 instability。IMPALA 的 V-trace 通过 truncated IS weight 缓解；GA3C 用 $\epsilon$-correction 防止 log probability 数值不稳定。

**Stale update 问题**：如图 6a 所示，Actor $j$ 在 $t_2$ 时刻用 $W_0$ 产生 data，但 target model 已更新到 $W_1$。red arrow 表示 stale gradient。

### 5.2 Synchronous On-policy Training（Fig. 6b）

所有 worker 完成后同步更新，保证 model consistency。两种实现：
- **Centralized**（PAAC, DBA3C）：gradient → Parameter Server → broadcast
- **Decentralized**（DD-PPO）：All-Reduce shuffle gradients

**Synchronization barrier**：fast worker 空闲等待 slow worker。在 heterogeneous cluster 中尤为严重。Adamski et al. 实验显示，64 nodes 上 synchronous on-policy 优于 asynchronous off-policy。

### 5.3 Stale-Synchronous Parallel (SSP)

Bounded staleness：允许 fast worker 领先，但最大 staleness 不超过阈值 $s_{\max}$：
$$
\|W_i^{(t)} - W_j^{(t)}\| \leq s_{\max} \cdot \eta
$$
在 DL 中已验证有效（Ho et al., NeurIPS 2013），但 DRL 中尚未广泛采用。paper 认为这是 great potential direction，可能基于 target policy 与 behavior policy 的 divergence 来 enforce synchronization。

### 5.4 On-policy + Off-policy 混合

Schmitt et al. (ICML 2020, [arXiv:1909.02087](https://arxiv.org/abs/1909.02087))：mix off-policy replay experience with on-policy data + trust region algorithm，超越 IMPALA。Borges et al. 在 MuZero 中 combine off-policy 与 on-policy targets。

---

## 6. Deep Evolutionary Reinforcement Learning

### 6.1 Evolution Strategies (ES)

Salimans et al. (2017, [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)) 的核心思想：在 parameter space 而非 action space 上搜索。

**目标函数**：
$$
\mathbb{E}_{\theta \sim p_\mu} F(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} F(\theta + \sigma \epsilon)
$$

其中：
- $\theta$：neural network parameters
- $p_\mu$：parameterized distribution（mean $\mu$，fixed covariance $\sigma^2 I$）
- $\sigma$：exploration scale（标准差）
- $\epsilon$：从标准正态分布 $\mathcal{N}(0, I)$ 采样的 noise vector
- $F(\theta)$：episode return

**Gradient estimator**（Eq. 2）：
$$
\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} F(\theta + \sigma \epsilon) = \frac{1}{\sigma} \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)} \left[ F(\theta + \sigma \epsilon) \cdot \epsilon \right]
$$

**直觉**：这个 estimator 的几何意义是——沿 $\epsilon$ 方向扰动 $\theta$，如果 return $F$ 增大，则 $\theta$ 应该朝 $\epsilon$ 方向移动；反之反方向移动。$\epsilon$ 起到 "directional sensor" 的作用。

**优势**：
1. **Black-box optimization**：invariant to action frequency 和 delayed rewards，适合 long horizon
2. **Massive parallelization**：worker 间只需传输 episode return（scalar），带宽需求极低
3. **No gradient synchronization**：无需 value function approximation 的额外同步

实验：80 machines + 1440 CPU cores，2 orders of magnitude speedup over single machine。

### 6.2 Genetic Algorithms (GA)

**Deep GA** (Such et al., 2017, [arXiv:1712.06567](https://arxiv.org/abs/1712.06567))：gradient-free。

**Mutation 操作**（Eq. 3）：
$$
\theta^n = \psi(\theta^{(n-1)}, \tau_n) = \theta^{(n-1)} + \sigma \epsilon(\tau_n)
$$

其中：
- $\theta^{(n-1)}$：parent generation 的 parameters
- $\theta^n$：child generation 的 parameters
- $\tau_n$：mutation seeds 列表
- $\epsilon(\tau_n)$：由 seeds 确定性生成的 Gaussian noise $\sim \mathcal{N}(0, \sigma)$
- $\sigma$：mutation strength

**关键技巧**：compact encoding——只存储 initialization seed + mutation seeds list，重建时按需 reconstruct parameters。这使得 distributed setting 下的通信开销极低。

**Truncation selection**：选 top-$T$ individuals 作为下一代 parents。

**NEAT** (Stanley & Miikkulainen, 2002, [DOI:10.1162/106365602320169811](https://doi.org/10.1162/106365602320169811))：同时 evolve network topology 和 weights，通过 augmenting topologies 逐步复杂化网络。

### 6.3 Hybrid: Evolution + Backpropagation

**Khadka & Tumer** (2019, [arXiv:1812.06567](https://arxiv.org/abs/1812.06567))（Fig. 7a）：
- Population of actor networks 用 GA 维持多样性
- DDPG 用 backpropagation 从 replay buffer 学习
- Learned weights 同步回 population

**GPO** (Gangwani & Peng, ICLR 2018)：crossover 在 state visitation space 而非 parameter space，避免破坏 hierarchical relationship。Mutation 用 PPO 而非 random permutation。

**Embodied Intelligence** (Gupta et al., Nature Communications 2021, [DOI:10.1038/s41467-021-25167-8](https://doi.org/10.1038/s41467-021-25167-8))（Fig. 7b）：1152 CPUs evolve 10 generations，4000 agent morphologies，每个 morphology 5M environment interactions。同时 evolve morphology 和 policy。

---

## 7. 实验数据对比分析

Table 1 提供了关键对比数据，我从几个维度提取 insight：

### 7.1 Speedup 趋势

| 方法 | 硬件 | Speedup | 启示 |
|------|------|---------|------|
| Gorila | 31 machines | 10× over GPU | Cluster computing baseline |
| A3C | 16 CPU cores | 2× over K40 GPU | Single machine 可比肩 cluster |
| GA3C | 16 CPU + 1 Titan X | 45× over A3C | GPU batch inference 关键 |
| SampleFactory | 36 CPU + 2080Ti | 4× over SEED_RL | 单机优化极限 |
| DD-PPO | 256 V100 GPU | 196× over 1 GPU | 同步 + All-Reduce 扩展性好 |
| Isaac Gym | 1 GPU | 300× over CPU cluster | Zero-copy GPU simulation 颠覆性 |
| SEED_RL | 520 CPU + 8 TPU v3 | 11× over IMPALA | TPU + centralized inference |

### 7.2 FPS（Frames Per Second）趋势

- Ape-X: 50K FPS（360 CPU cores）
- Isaac Gym: 300K+ FPS（单 GPU）
- Brax: hundreds of millions steps/sec（TPUv3 8×8）

**直觉**：从 50K 到 hundreds of millions，跨越 4 个数量级，核心驱动力是 **simulation 从 CPU 迁移到 GPU/TPU 内部 memory**，消除 off-chip communication。

### 7.3 训练时间压缩

- DQN (2015): 38 days on Atari
- A3C (2016): ~19 days
- Ray RLlib: 3.7 minutes for Atari
- SpeedyZero: 35 minutes mastering Atari with only 300K samples

---

## 8. Open-Source Libraries 对比

Table 2 比较了 16 个库。关键维度：

### 8.1 Algorithm 覆盖度

- **RLlib** (Berkeley, [rllib.io](https://rllib.io/))：最全，含 A2C, A3C, ARS, DDPG, TD3, Rainbow, ES, APE-X, IMPALA, PPO, APPO, DD-PPO, SAC, QMIX, VDN, IQN, MADDPG, MCTS
- **rlpyt** (Berkeley)：A2C, PPO, DQN variants, Rainbow, R2D2, Ape-X, DDPG, TD3, SAC
- **TorchRL** (UPF, 2023)：A2C, PPO, DDQN, SAC, REDQ, Dreamer, Decision Transformers, RLHF, APPO, DPPO, DDPPO, IMPALA, Ape-X

### 8.2 并行特性

- **Ray RLlib**：synchronous + straggler migrations，cluster automation on AWS/Azure/GCP
- **SEED_RL**：gRPC-based fast communication layer，centralized inference
- **Fiber** (Uber)：standard multiprocessing API，可在单机与 cluster 间无缝迁移
- **SampleFactory**：单机异步训练 + off-policy corrections，支持 IsaacGym/Brax/Envpool
- **SRL**：synchronous training + data batch prefetching，10K+ cores 扩展
- **PEARL**：FPGA 跨平台 portable implementation

### 8.3 环境支持

SampleFactory 支持最广（10 类）：MuJoCo, Atari, VizDoom, DeepMind Lab, Megaverse, Envpool, IsaacGym, Brax, Quad-Swarm-RL, HuggingFace Hub。

---

## 9. Future Directions 深度分析

### 9.1 Specialized Hardware Accelerators

- **In-memory computing**：减少 data movement，[Nature Nanotechnology 2020](https://doi.org/10.1038/s41565-020-0655-z)
- **Pipeline parallelism on hardware**：保证所有 array 始终 active
- **Neuromorphic hardware**：spike-based chips，[Nature Machine Intelligence 2022](https://doi.org/10.1038/s42256-022-00480-x)
- **Heterogeneous CPU-FPGA-GPU-TPU** 组合

### 9.2 In-network Distributed Aggregation

Li et al. (ISCA 2019, [DOI:10.1145/3352460.3358299](https://doi.org/10.1145/3352460.3358299)) 开创 in-switch computing for distributed DRL。核心思路：gradient aggregation 在 programmable switch 上 on-the-fly 完成，以 packet 为粒度而非 vector 为粒度，大幅减少 end-to-end latency 和 network traffic。

### 9.3 Efficient Sample Exploitation

OpenAI Five 的 sample reuse rate < 1（异步训练）。改进方向：
- **Priority-refresh replay** (SpeedyZero)
- **Asynchronous curriculum experience replay** ([IEEE TVT 2023](https://doi.org/10.1109/TVT.2023.3290033))
- **Regret minimization replay** ([NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/18c4dcd4e34e5957f0e4e4c5e5e5e5e5-Abstract.html))
- **Curiosity-driven exploration** ([CoRL 2023](https://corl2023.org/))
- **Diversity-driven + Novelty search**

### 9.4 LLM-enhanced DRL

LLM 在 DRL 训练中的三个作用：
1. **Reward function design**：LLM 生成 reward shaping
2. **Action selection**：LLM 提供 prior knowledge 指导探索
3. **Policy evaluation**：LLM 评估当前 policy 质量

代表工作：
- Constitutional AI / RLHF ([Bai et al., 2022](https://arxiv.org/abs/2204.05862))
- LLM feedback for robotic manipulation ([Chu et al., 2023](https://arxiv.org/abs/2311.02379))
- LLM-enhanced RL survey ([Cao et al., 2024](https://arxiv.org/abs/2404.00282))

---

## 10. Intuition 总结与个人思考

从这篇 survey 中可以提炼出 DRL distributed training 的几条核心 design principle：

### 10.1 Data Locality 优于 Raw Parallelism

Isaac Gym 的 300× speedup 本质上是把 simulation、inference、training 全部放在同一 GPU memory 内，消除 PCIe bottleneck。这印证了系统设计中 "data locality is king" 的原则。未来 in-memory computing 和 neuromorphic chips 会进一步强化这一趋势。

### 10.2 Synchronization Granularity 是关键 Trade-off

| 机制 | Consistency | Resource Utilization | 适用场景 |
|------|-------------|---------------------|----------|
| Async off-policy | Low | High | Off-policy, 大规模 |
| Sync on-policy | High | Low | On-policy, 中等规模 |
| Stale-sync | Medium | Medium | 未充分探索 |

DRL 特有的 challenge 是 policy-lag——behavior policy 与 target policy 的 divergence。一个有趣的 intuition 是：**staleness bound 应该与 policy entropy 相关**——高 entropy policy 可以容忍更大 staleness，低 entropy policy 需要更严格同步。

### 10.3 Evolution 提供互补的 Optimization Landscape

ES/GA 的价值在于：
1. **Gradient-free**：避开 backpropagation 的 local optima 问题
2. **Inherently parallel**：population-based 天然适合 distributed
3. **Exploration diversity**：population 维持多种行为策略

但 sample complexity 高。Hybrid 方法（evolution + backprop）是 promising direction——evolution 提供 diverse starting points，backprop 提供 efficient local optimization。

### 10.4 Heterogeneity 是 First-class Concern

Real cluster 的 heterogeneity（CPU/GPU/FPGA/TPU 混合、不同速度 worker）使得：
- Preemptive scheduling 必不可少（DD-PPO 的 straggler preemption）
- Load balancing 需要动态（Ray 的 fine/coarse-grained 两级）
- Computation-communication overlap 是 throughput 关键（PEARL）

### 10.5 DRL vs DL 的 Distributed Training 差异

| 维度 | DL | DRL |
|------|----|----|
| Data source | Static dataset | Dynamic environment interaction |
| Data distribution | Fixed | Shifts with policy update |
| Worker roles | Homogeneous | Heterogeneous (Actor/Learner/PS) |
| Synchronization | Mostly synchronous | Async + sync + stale-sync |
| Bottleneck | Gradient communication | Simulation + communication |

这个对比解释了为什么不能直接 copy DL 的 distributed methods 到 DRL——environment interaction 引入的时间维度使得 staleness 与 policy-lag 成为 first-class problem。

---

## 参考 Web Links

1. Gorila: https://arxiv.org/abs/1507.04296
2. A3C: https://arxiv.org/abs/1602.01783
3. Ape-X: https://arxiv.org/abs/1803.00933
4. R2D2: https://arxiv.org/abs/1802.01561
5. IMPALA: https://arxiv.org/abs/1802.01594
6. DD-PPO: https://arxiv.org/abs/1911.00357
7. SEED_RL: https://arxiv.org/abs/1910.06591
8. Isaac Gym: https://arxiv.org/abs/2108.10470
9. Brax: https://arxiv.org/abs/2106.13281
10. CuLE: https://arxiv.org/abs/1907.08424
11. ES (Salimans): https://arxiv.org/abs/1703.03864
12. Deep GA: https://arxiv.org/abs/1712.06567
13. Evolution-Guided PG: https://arxiv.org/abs/1812.06567
14. SampleFactory: https://arxiv.org/abs/2002.12444
15. Ray RLlib: https://arxiv.org/abs/1712.09381
16. TorchRL: https://arxiv.org/abs/2306.00377
17. Embodied Intelligence: https://www.nature.com/articles/s41467-021-25167-8
18. In-switch computing (ISCA 2019): https://doi.org/10.1145/3352460.3358299
19. Gossip Actor-Learner: https://arxiv.org/abs/1906.04585
20. SRL (ICLR 2024): https://arxiv.org/abs/2402.19373

这篇 survey 的核心贡献在于建立了一个清晰的 taxonomy：从 system architecture → simulation parallelism → computing parallelism → synchronization mechanism → evolution-based training，层层递进。对于想进入 distributed DRL 领域的研究者，它提供了从 baseline 算法到工程实现的完整 roadmap。最 promising 的 future direction 我认为是 **LLM-enhanced DRL** 与 **in-network aggregation** 的结合——LLM 提供高质量 prior knowledge 减少 sample 需求，in-network computing 减少 communication overhead，两者协同可能突破当前 DRL 的 efficiency ceiling。
