# 文章概览：将 Kubernetes 迁移到 Hetzner 的实战经验

这篇文章记录了作者将 **Kubernetes cluster** 从 **DigitalOcean** 迁移到 **Hetzner** 的完整过程，核心成果是：

> **基础设施成本降低 75%，计算资源翻倍**

---

## 1. 核心动机与背景

### 为什么选择 Hetzner？

| 对比项 | Hetzner | AWS/DigitalOcean |
|--------|---------|------------------|
| 价格 | 极低 | 较高 |
| 托管 K8s | ❌ 无 | ✅ 有 (但节点价格 +100%) |
| 裸金属服务器 | ✅ 有 | ✅ 有限/昂贵 |
| 功能丰富度 | 基础 | 全面 |

**第一性原理分析**：如果你的 workload 有**稳定的最小容量需求**，自管理 control plane 的成本分摊后远比托管方案划算。公式如下：

$$\text{自管成本} = \$20/\text{月 (control plane)} + \text{运维时间}$$

$$\text{托管溢价} = N_{\text{nodes}} \times P_{\text{node}} \times 100\%$$

当 $N_{\text{nodes}} \geq 6$ 时，年节省已覆盖 control plane 成本；100 个节点时，托管方案相当于**双倍 VM 成本**。

---

## 2. 架构设计细节

### 2.1 Node Roles 与命名规范

```
<region><zone><environment><role><number>
```

例如：`euc1pmgr1` = **Eu**rope **c**entral, **1**st zone, **p**roduction, **mgr** (control-plane), **1**

作者只保留三个 role：
- **control-plane**：运行 etcd、kube-apiserver、kube-scheduler、kube-controller-manager
- **worker**：运行应用 pods
- **database**：运行数据库（使用 local SSD）

> **最小化 role 数量** → 简化 scheduling → 提高 resource utilization

### 2.2 Infrastructure as Code

```
Terraform (基础设施编排)
    ↓
    创建 Server + Firewall + Network
    ↓
Puppet (配置管理)
    ↓
    安装 Kubernetes 组件 + 加入集群
```

**关键 Terraform 资源**：
```hcl
resource "hcloud_firewall" "controlplane" {
  name = "control-plane"
  apply_to { label_selector = "role=control-plane" }
}

resource "hcloud_placement_group" "controlplane" {
  type = "spread"  # 确保节点物理分散
}
```

**Placement Group 的作用**：Hetzner 的 `spread` 类型确保同一 placement group 内的服务器部署在不同物理主机上，避免单点故障。

---

## 3. 网络架构深度解析

### 3.1 多层网络隔离

```
┌─────────────────────────────────────────────────────────────┐
│                     Internet                                 │
│                         ↓                                    │
│              CloudFlare (CDN + DNS)                         │
│                         ↓                                    │
│           Hetzner Load Balancer (~$5/month)                 │
│                         ↓                                    │
│              Nginx Ingress Controller                        │
│                         ↓                                    │
│    ┌───────────────────────────────────────────┐            │
│    │        Calico CNI (Pod Network)            │            │
│    │  Pod CIDR: overlay/IP-in-IP                │            │
│    └───────────────────────────────────────────┘            │
│                         ↓                                    │
│    ┌───────────────────────────────────────────┐            │
│    │   Hetzner Private Network (10.0.0.0/16)    │            │
│    └───────────────────────────────────────────┘            │
│                         ↓                                    │
│    ┌───────────────────────────────────────────┐            │
│    │   Tailscale VPN (管理访问)                  │            │
│    │   WireGuard + NAT Hole Punching            │            │
│    └───────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Tailscale 的巧妙应用

**传统方案**：开放 SSH 端口 (22) → 安全风险

**Tailscale 方案**：
- 所有节点出站连接到 Tailscale coordination server
- **NAT Hole Punching**：双向出站连接建立对等隧道
- 从外部扫描：**零开放端口**

> 引用原文："An attacker doing a port scan will see no openings at all, because the firewall is not open to them at all."

### 3.3 Calico 与 Tailscale 冲突解决

**问题**：两者默认使用相同的 netfilter mark mask：
```
Tailscale: 0xff00ff00
Calico:    0xff00ff00
```

导致 Tailscale 的 iptables 规则错误应用于 inter-pod 流量。

**解决方案**：
```yaml
- name: FELIX_IPTABLESMARKMASK
  value: "0xff00ff00"  # 实际上是改为不同的值
```

**技术原理**：Netfilter mark 是一个 32-bit 数值，用于给数据包打标签，不同子系统需要使用不同的 bit mask 避免冲突。

### 3.4 Cloud Controller Manager

Hetzner 提供的 [hcloud-cloud-controller-manager](https://github.com/hetznercloud/hcloud-cloud-controller-manager) 实现：

1. **自动配置 Node External-IP**
2. **自动创建/删除 Load Balancer**（通过 Service annotations）
3. **Network 资源管理**

kubelet 启动参数：
```bash
--cloud-provider=external
```

---

## 4. 存储架构：Cloud Volume vs Local SSD

### 4.1 Hetzner Cloud Volumes 的问题

作者实测数据：

| 指标 | Hetzner Volume | AWS EBS (gp3) |
|------|----------------|---------------|
| IOWAIT | >50ms | <10ms |
| IOPS | 低 | 高 (16,000) |
| 适用场景 | 日志、轻量存储 | 数据库 |

**原因分析**：Hetzner Volume 是网络块存储，经过虚拟化层 + 网络传输，延迟累积。

### 4.2 Local Static Provisioner 方案

```
┌────────────────────────────────────────────────────────┐
│                    Physical Node                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  /dev/sda (NVMe SSD)                              │  │
│  │      ↓                                            │  │
│  │  LUKS Encryption (dm-crypt)                       │  │
│  │      ↓                                            │  │
│  │  LVM (Logical Volume Manager)                     │  │
│  │  ├── /dev/vg-data/lv-small                        │  │
│  │  ├── /dev/vg-data/lv-medium                       │  │
│  │  └── /dev/vg-data/lv-large                        │  │
│  │      ↓                                            │  │
│  │  Local Static Provisioner                         │  │
│  │      ↓                                            │  │
│  │  PersistentVolume (PV)                            │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**LVM 的优势**：
- 动态调整卷大小（但 K8s 中不推荐）
- 快照支持（LVM snapshot）
- 条带化提高性能

**Local PV 的代价**：
- Pod 与 Node 强绑定
- Node 故障时，PV 需手动清理
- 不适合 ReadWriteMany (RWX) 场景

**作者选择的数据库方案**：

| 数据库 | 高可用方案 | 存储类型 |
|--------|-----------|---------|
| PostgreSQL | Synchronous Replication | Local PV |
| Redis | Sentinel | Local PV |

### 4.3 CSI Driver 配置示例

```yaml
storageClasses:
  - name: snapshotted-volumes
    defaultStorageClass: true
    reclaimPolicy: Retain  # 防止误删
  - name: snapshotted-luks
    parameters:
      csi.storage.k8s.io/node-publish-secret-name: encryption-secret
```

**注意**：Hetzner 目前**不支持 Volume Snapshot**（这是与 AWS EBS 的主要差距）。

---

## 5. 备份策略：1-2-3 原则

```
1-2-3 Backup Rule:
├── 3 copies of data
├── 2 different media types
└── 1 offsite copy

作者实现:
├── Hetzner Cloud Volume (本地)
├── S3 Provider A (异地)
└── S3 Provider B (异地)
```

### 5.1 Velero：K8s 原生备份

```yaml
# Pod annotation 触发备份
pod.annotations: 
  backup.velero.io/backup-volumes: "data-volume"
```

**Velero 备份内容**：
- Kubernetes API objects (Deployments, Services, ConfigMaps...)
- Persistent Volume 数据

### 5.2 pgBackRest：PostgreSQL 专用

```
PostgreSQL
    ↓
pgBackRest
    ├── Full Backup (每周)
    ├── Differential Backup (每日)
    └── WAL Archiving (持续)
            ↓
    S3 + Hetzner Volume
```

**关键概念**：
- **WAL (Write-Ahead Log)**：PostgreSQL 的重做日志
- **Point-in-Time Recovery (PITR)**：恢复到任意时间点

---

## 6. 监控栈

```
Node Exporter (metrics) ─┐
                         ├──→ Prometheus ──→ Grafana
Application metrics ─────┘         │
                                   ↓
                           Alertmanager ──→ Slack
                                   ↑
Loki (logs) ───────────────────────┘
```

---

## 7. 遇到的坑与解决方案

### 7.1 IP 质量问题（最大坑）

**现象**：约 10% 的 Hetzner IP 被 Google Cloud 封锁

**原因**：
1. 部分 IP 的 GeoIP 数据库错误标记为伊朗
2. Hetzner 低价吸引 VPN 用户绕过制裁
3. Scraping bots 滥用

**解决方案**：
```bash
# 方案一：WireGuard 出口代理
# 方案二：Floating IP（长期方案）
# 方案三：接受被封 IP，使用 Tailscale 路由
```

**影响范围**：所有 GCP 服务（GitLab CI、Sentry、Incident.io）

### 7.2 数据中心延迟问题

| 路径 | 延迟 |
|------|------|
| Nuremberg ↔ Falkenstein | ~3ms |
| Nuremberg ↔ Helsinki | ~25ms |

**影响**：
- Synchronous PostgreSQL Replication 写入延迟增加
- etcd 不受影响（默认 heartbeat 100ms）

**作者决策**：
```
Active-Active: Nuremberg + Falkenstein
Batch Workloads: Helsinki
```

### 7.3 C-Lion1 海底光缆事件

2024年11月18日，德国-芬兰唯一海底光缆被切断，Hetzner 报告 backbone 问题 10 天。

---

## 8. Hetzner 的局限与权衡

| 方面 | Hetzner | 影响 |
|------|---------|------|
| 托管 K8s | ❌ | 需自管 control plane |
| Volume Snapshot | ❌ | 依赖应用层备份 |
| S3 | 新推出 (2024.12 GA) | 早期不稳定 |
| 客服 | 德式风格 | 1-2 工作日响应 |
| IP 信誉 | 部分问题 | 需 workaround |

---

## 9. 总结：适合迁移到 Hetzner 的场景

**✅ 适合**：
- 成本敏感型项目
- 有 Kubernetes 运维能力
- Workload 对托管服务依赖低
- 可以接受部分 IP 质量问题

**❌ 不适合**：
- 需要 24/7 企业级支持
- 依赖 Volume Snapshot
- 无法接受自管 control plane
- 对 IP 信誉要求极高（邮件发送等）

---

## 关键技术参考链接

- [Hetzner Cloud Controller Manager](https://github.com/hetznercloud/hcloud-cloud-controller-manager)
- [Hetzner CSI Driver](https://github.com/hetznercloud/csi-driver)
- [Tailscale](https://tailscale.com/)
- [Calico CNI](https://www.tigera.io/project-calico/)
- [Velero Backup](https://velero.io/)
- [pgBackRest](https://pgbackrest.org/)
- [Local Static Provisioner](https://github.com/kubernetes-sigs/sig-storage-local-static-provisioner)
- [SealedSecrets](https://github.com/bitnami-labs/sealed-secrets)
- [CrunchyData Postgres Operator](https://github.com/CrunchyData/postgres-operator)

---

这篇文章的核心价值在于：**用第一性原理分析云服务成本结构**，证明了对于稳定工作负载，低成本云 + 自管基础设施的组合在经济性和技术可控性上都有优势，但需要承担运维复杂度和部分功能缺失的代价。