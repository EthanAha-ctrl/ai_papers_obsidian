从 First Principles 出发，compute 的本质是执行 instruction 与处理 data。在早期，application 直接运行在 bare metal 上，但是 dependency 冲突 与 resource 争抢 导致了 isolation 的需求。于是人类发明了 Virtual Machine (VM)，通过 Hypervisor 引入 hardware-level 的 isolation，但是 VM 包含了完整的 Guest OS，overhead 极大。为了追求更轻量的 isolation，Container 诞生了（例如 Docker），它利用 Linux Kernel 的 Namespace 与 Cgroups，实现了 process-level 的 isolation。

**但是，Container 解决的是 single machine 上的 execution 问题，并没有解决 distributed system 上的 lifecycle 问题。** 当你有 10,000 个 container 运行在 1,000 台 machine 上时，如果 machine 宕机、traffic 突增、或者需要 rolling update，手动管理是不可能的。这就是 Kubernetes (K8s) 诞生的 First Principle 动机：**Kubernetes 是 Cloud 的 Operating System，它将底层 fragmented 的 compute resources 抽象为一个统一的 resource pool，并自动化管理 container 的 lifecycle。**

---

### Architecture Parsing (架构图解析)

Kubernetes 遵循 Client-Server 架构，分为 Control Plane (Master) 与 Worker Node。

#### 1. Control Plane (大脑)
Control Plane 负责 global decision making 与 detecting cluster state changes。
*   **kube-apiserver**: 唯一的 entry point。所有 component 之间的通信都通过 REST API 进行。它负责 authentication, authorization, admission control。
*   **etcd**: 一致且高可用的 distributed key-value store。它是 cluster 的 "single source of truth"。所有 cluster state data 都存储在这里。使用 Raft consensus algorithm 保证 consistency。
*   **kube-scheduler**: 负责 Pod 的 placement。它过滤不符合条件的 Node，然后对剩余 Node 打分，选择得分最高的 Node。
*   **kube-controller-manager**: 运行 controller process 的 daemon。核心逻辑是 **Reconciliation Loop (调和循环)**，不断观测 actual state，并驱动其向 desired state 收敛。

#### 2. Worker Node (手脚)
Node 是实际运行 workload 的 machine。
*   **kubelet**: 运行在每个 Node 上的 agent。它接收 PodSpec，确保 Container Runtime 启动并健康运行 container。
*   **kube-proxy**: 维护 Node 上的 network rules。利用 iptables 或 IPVS 实现 Service 的 load balancing 与 routing。
*   **Container Runtime**: 真正运行 container 的软件（如 containerd, CRI-O），通过 CRI (Container Runtime Interface) 与 kubelet 交互。

---

### Technical Deep Dive (更细节的技术讲解与公式)

#### 1. Reconciliation Loop (核心控制逻辑)
Kubernetes 是 declarative system。你声明 "I want 3 replicas of Nginx"，controller 会自动帮你达成。

我们可以用数学公式来表达这个 Reconciliation 过程。定义 $E_t$ 为时间 $t$ 的 Error State，$S_d$ 为 Desired State，$S_a(t)$ 为 Actual State at time $t$。

$$ E_t = || S_d - S_a(t) ||^2 $$

Controller 的目标是使得 $E_t \to 0$。每次 reconciliation loop 就是一个 gradient descent step：

$$ S_a(t + \Delta t) = S_a(t) + \alpha \nabla_{S_a} (-E_t) $$

*   $E_t$: Error function (二次方范数，衡量偏差程度)。
*   $S_d$: Desired state (例如 `replicas: 3`)。
*   $S_a(t)$: Actual state (例如当前运行的 healthy pod 数量为 2)。
*   $\alpha$: Reconciliation rate (控制循环的频率，避免系统震荡)。
*   $\Delta t$: Loop 的执行间隔。
*   $\nabla_{S_a}$: 针对 actual state 的梯度，即需要采取的 action（在这个例子中，action 是创建 1 个新 pod）。

#### 2. Scheduling Algorithm (调度算法解析)
当 Pod 被创建但未分配 Node 时，`kube-scheduler` 会执行两阶段算法：

**Phase 1: Filtering (过滤)**
排除不满足 Pod 硬性需求的 Node。例如 Pod 需要 4 CPU，而 Node 只有 2 CPU，则被过滤。
公式化表示，Node 集合 $N_{feasible}$ 为：

$$ N_{feasible} = \{ n \in N_{cluster} \mid \forall p \in Predicates, p(n, Pod) = True \} $$

*   $N_{cluster}$: Cluster 中所有的 Node 集合。
*   $Predicates$: 硬性过滤条件集合（如 ResourceFit, PodAntiAffinity）。
*   $p(n, Pod)$: 判断 Node $n$ 是否满足 Pod 条件的布尔函数。

**Phase 2: Scoring (打分)**
对 $N_{feasible}$ 中的 Node 进行打分，选择最优解。打分是多个 Priority Function 的加权和。

$$ Score(n) = \sum_{i=1}^{k} w_i \cdot f_i(n, Pod) $$

*   $Score(n)$: Node $n$ 的最终得分。
*   $k$: Scoring plugins 的数量。
*   $w_i$: 第 $i$ 个 plugin 的 weight (权重)。
*   $f_i(n, Pod)$: 第 $i$ 个 plugin 对 Node $n$ 的打分函数（归一化到 [0, 100]）。例如 `NodeResourcesFit` plugin 会倾向于选择 Resource allocation 最均衡的 Node，即 $(1 - \frac{requested\_cpu}{capacity\_cpu})$ 比例较高的 Node。

#### 3. Resource Management: Requests vs Limits
Kubernetes 管理 CPU 与 Memory 的方式不同，这基于它们的物理特性。

*   **CPU** 是 **Compressible Resource**。当 Node 上 CPU 资源不足时，container 会被 throttled (限速)，但不会被 killed。时间片被拉长。
*   **Memory** 是 **Incompressible Resource**。当 Node 内存耗尽，kernel 的 OOM (Out of Memory) killer 会被触发，杀掉占用内存最多的 process。

Kubernetes 定义了 QoS (Quality of Service) Classes：
1.  **Guaranteed**: `requests == limits` (CPU 与 Memory)。最高优先级，最后被 OOM Kill。
2.  **Burstable**: `requests < limits`。中等优先级。在内存不足时，如果使用量超过 `requests`，有被杀的风险。
3.  **BestEffort**: 没有 `requests` 与 `limits`。最低优先级，系统内存紧张时第一个被杀。

OOM Score 计算启发式公式 (Linux Kernel 机制):
$$ oom\_score\_adj = -1000 + \left\lfloor \frac{1000}{1 + \frac{Memory\_Limit}{Memory\_Usage}} \right\rfloor $$
*值越低越不容易被杀。Guaranteed Pod 通常被设置为 -998，而 BestEffort 被设置为 1000。*

---

### Experimental Data Table (实验数据模拟)

下面是一个典型的 Kubernetes Cluster 在进行 Rolling Update (滚动更新) 时的性能数据模拟。Cluster 包含 100 个 Node，运行 500 个 Pods (每个 Pod 1 CPU Request, 1GB Memory Request, Limits 为 2 CPU, 2GB Memory)。

| Observation Time (s) | Desired Replicas | Running Replicas | Available Replicas | API Server Latency (ms) | Scheduling Latency (ms) | Node CPU Utilization (%) | Node Memory Utilization (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| T=0 (Stable) | 500 | 500 | 500 | 12 | N/A | 65% | 70% |
| T=10 (Trigger Update) | 500 | 500 | 500 | 18 | 15 | 66% | 70% |
| T=20 (Surge 25%) | 525 | 525 | 510 | 45 | 120 | 82% | 78% |
| T=30 (Terminating old)| 525 | 510 | 505 | 25 | 40 | 75% | 74% |
| T=40 (Surge 25%) | 525 | 525 | 515 | 50 | 135 | 85% | 79% |
| T=60 (Converging) | 500 | 500 | 500 | 15 | 20 | 68% | 71% |
| T=120 (Stable) | 500 | 500 | 500 | 11 | N/A | 64% | 69% |

**Data Analysis:**
在 T=20 与 T=40 阶段，由于 `maxSurge=25%` 的策略，新 Pod 被创建，导致 Running Replicas 超过 Desired，此时 API Server 与 Scheduler Latency 显著上升（因为 etcd 事务增加，Scheduler 需要重新计算 Node Scores）。同时 Node Utilization 出现峰值，这验证了 Rolling Update 需要预留 Resource Buffer，否则可能因 CPU Throttling 导致 application 降级。

---

### Broad Associations & Ecosystem (发散联想)

1.  **Network Layer (CNI)**: Kubernetes 自身不实现 network。它通过 CNI (Container Network Interface) 插件（如 Calico, Cilium）实现 Pod 间的通信。最新的趋势是使用 **eBPF** 技术绕过 Linux Kernel 的 iptables，直接在 kernel 层面处理 network packet，将 latency 从毫秒级降低到微秒级。
2.  **Storage Layer (CSI)**: Container 是 ephemeral 的。CSI (Container Storage Interface) 允许云厂商（如 AWS EBS, Ceph）提供 persistent volume，使得 data 可以独立于 Pod lifecycle 存在。
3.  **Service Mesh**: 当 Microservices 架构极度复杂时，Kubernetes 内部的 Service 机制（kube-proxy）无法满足 granular traffic control (如 circuit breaking, mutual TLS)。Service Mesh (如 Istio, Linkerd) 通过在每个 Pod 中注入 Sidecar Proxy (Envoy)，intercept 所有 network traffic，实现了 application logic 与 network logic 的 decoupling。

---

### Web Links for Reference

1.  **Kubernetes Official Documentation (The definitive source):**
    https://kubernetes.io/docs/concepts/architecture/
2.  **Google Borg/Omega/Kubernetes Lineage Paper (Historical First Principles of K8s):**
    https://research.google/pubs/pub43438/ (Large-scale cluster management at Google with Borg)
3.  **Cilium & eBPF Networking (Advanced intuition on modern K8s networking):**
    https://cilium.io/blog/2021/11/29/cni-benchmark-evolution
4.  **Kubernetes Scheduling Framework Deep Dive:**
    https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/
5.  **Kubernetes QoS and OOM Management:**
    https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/