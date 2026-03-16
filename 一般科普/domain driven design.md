# Domain-Driven Design (DDD) 全面解析

## 一、DDD 的核心哲学与基本概念

Domain-Driven Design (DDD) 是一种软件设计方法论，由 **Eric Evans** 于 2003 年在其经典著作 *Domain-Driven Design: Tackling Complexity in the Heart of Software* 中提出。DDD 的核心思想是：**将软件模型与业务领域紧密结合，使代码的语言和结构与业务领域的语言和结构保持一致**。

### 1.1 DDD 的三大支柱

| 支柱 | 英文名称 | 核心思想 |
|------|----------|----------|
| 统一语言 | **Ubiquitous Language** | 开发者、领域专家、用户使用相同的术语交流 |
| 战略设计 | **Strategic Design** | 从宏观层面划分系统边界，定义 Bounded Context 之间的关系 |
| 战术设计 | **Tactical Design** | 在微观层面构建领域模型的技术模式 |

---

## 二、Strategic Design：宏观架构视角

### 2.1 Domain 与 Subdomain

一个复杂的业务领域 可以分解为多个子领域：

```
Domain (领域)
│
├── Core Subdomain (核心子领域)
│   └── 企业核心竞争力所在，需要最精细的设计
│
├── Supporting Subdomain (支撑子领域)
│   └── 支撑核心业务，但非差异化竞争点
│
└── Generic Subdomain (通用子领域)
    └── 通用解决方案，可购买现成软件
```

**公式化理解**：

$$\text{Domain} = \bigcup_{i=1}^{n} \text{Subdomain}_i$$

其中：
- $n$ = 子领域的数量
- $\text{Subdomain}_i$ = 第 $i$ 个子领域
- $\bigcup$ = 并集运算，表示所有子领域共同构成完整领域

### 2.2 Bounded Context（限界上下文）

**Bounded Context** 是 DDD 中最核心的概念之一。它定义了模型的一致性边界，在边界内部，每个术语都有明确、唯一的含义。

```
┌─────────────────────────────────────────────────────────────────┐
│                        E-Commerce System                         │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Catalog    │    │   Ordering   │    │   Shipping   │       │
│  │  Context     │    │   Context    │    │   Context    │       │
│  │              │    │              │    │              │       │
│  │ "Product" =  │    │ "Product" =  │    │ "Product" =  │       │
│  │ 商品信息     │    │ 订单中的商品 │    │ 待发货物品   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                      Context Mapping                            │
└─────────────────────────────────────────────────────────────────┘
```

**关键公式 - Bounded Context 边界约束**：

$$\forall t \in \text{BoundedContext}_i : \text{Meaning}(t) = \text{Unique}(t)$$

其中：
- $t$ = 任意术语
- $\text{Meaning}(t)$ = 术语 $t$ 的含义
- $\text{Unique}(t)$ = 在该上下文内的唯一解释

### 2.3 Context Mapping Patterns（上下文映射模式）

根据 Eric Evans 的定义，不同 Bounded Context 之间存在多种关系模式：

| 模式 | 英文名 | 适用场景 |
|------|--------|----------|
| 合作关系 | **Partnership** | 两个团队荣辱与共，需要协调开发 |
| 共享内核 | **Shared Kernel** | 共享一小部分领域模型代码 |
| 客户/供应商 | **Customer/Supplier** | 上下游关系，下游是客户 |
| 遵奉者 | **Conformist** | 下游团队无法影响上游，只能遵从 |
| 防腐层 | **Anticorruption Layer** | 保护下游模型不被上游污染 |
| 开放主机服务 | **Open-Host Service** | 提供标准协议供多个系统集成 |
| 发布语言 | **Published Language** | 使用行业标准作为通信媒介 |
| 分道扬镳 | **Separate Ways** | 完全独立，无集成需求 |
| 大泥球 | **Big Ball of Mud** | 无法识别边界的遗留系统 |

**Anticorruption Layer 架构图**：

```
┌────────────────────┐         ┌─────────────────────────────┐
│   Upstream         │         │   Anticorruption Layer      │
│   Context          │         │                             │
│                    │         │  ┌───────────────────────┐  │
│   ┌────────────┐   │         │  │  Adapter              │  │
│   │ External   │───┼────────▶│  │  (Model Translation) │  │
│   │ Model      │   │         │  └───────────────────────┘  │
│   └────────────┘   │         │            │                │
└────────────────────┘         │            ▼                │
                               │  ┌───────────────────────┐  │
                               │  │   Internal Model      │  │
                               │  │   (Protected)         │  │
                               │  └───────────────────────┘  │
                               └─────────────────────────────┘
```

---

## 三、Tactical Design：微观模型构建

### 3.1 Entity（实体）vs Value Object（值对象）

#### Entity（实体）

**Entity** 是通过**身份标识** 而非属性来定义的对象。

```
┌─────────────────────────────────────────┐
│              Entity: Customer           │
├─────────────────────────────────────────┤
│  - id: CustomerId (身份标识)             │
│  - name: String                         │
│  - email: String                        │
│  - address: Address                     │
├─────────────────────────────────────────┤
│  + changeEmail(newEmail: String)        │
│  + updateAddress(newAddress: Address)   │
└─────────────────────────────────────────┘
```

**身份恒等性公式**：

$$\forall e_1, e_2 \in \text{Entity} : e_1 = e_2 \iff \text{Id}(e_1) = \text{Id}(e_2)$$

其中：
- $e_1, e_2$ = 两个实体实例
- $\text{Id}(e)$ = 实体 $e$ 的唯一标识符
- $\iff$ = 当且仅当

#### Value Object（值对象）

**Value Object** 是不可变对象，通过**属性值** 来定义，没有身份标识。

```
┌─────────────────────────────────────────┐
│         Value Object: Address          │
├─────────────────────────────────────────┤
│  - street: String (不可变)              │
│  - city: String (不可变)                │
│  - zipCode: String (不可变)             │
│  - country: String (不可变)             │
├─────────────────────────────────────────┤
│  + equals(other: Address): Boolean     │
│  + withCity(newCity: String): Address   │
└─────────────────────────────────────────┘
```

**值相等性公式**：

$$\forall v_1, v_2 \in \text{ValueObject} : v_1 = v_2 \iff \bigwedge_{i=1}^{n} \text{Attr}_i(v_1) = \text{Attr}_i(v_2)$$

其中：
- $v_1, v_2$ = 两个值对象实例
- $\text{Attr}_i$ = 第 $i$ 个属性
- $n$ = 属性总数
- $\bigwedge$ = 逻辑与，所有属性都相等时两个值对象才相等

**不可变性原则**：

$$\text{Mutate}(v, \text{attr}, \text{value}) \rightarrow v' \text{ where } v' = \text{copy}(v) \land \text{set}(v', \text{attr}, \text{value})$$

### 3.2 Aggregate（聚合）与 Aggregate Root（聚合根）

**Aggregate** 是一组相关对象的集合，作为数据修改的单元。**Aggregate Root** 是聚合的根实体，外部只能通过它访问聚合内部对象。

```
┌────────────────────────────────────────────────────────────┐
│                    Aggregate: Order                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Aggregate Root: Order                        │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ - id: OrderId                                   │  │  │
│  │  │ - customerId: CustomerId                        │  │  │
│  │  │ - orderItems: List<OrderItem>                   │  │  │
│  │  │ - status: OrderStatus                           │  │  │
│  │  │ - totalAmount: Money                            │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ + addItem(product, quantity)                    │  │  │
│  │  │ + removeItem(orderItemId)                        │  │  │
│  │  │ + submit()                                       │  │  │
│  │  │ + cancel()                                       │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                        │                              │  │
│  │                        │ 控制                         │  │
│  │         ┌──────────────┼──────────────┐              │  │
│  │         ▼              ▼              ▼              │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐        │  │
│  │  │OrderItem  │  │OrderItem  │  │ Money     │        │  │
│  │  │(Entity)   │  │(Entity)   │  │(ValueObj) │        │  │
│  │  └───────────┘  └───────────┘  └───────────┘        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  边界：外部只能引用 Order，不能直接引用 OrderItem          │
└────────────────────────────────────────────────────────────┘
```

**聚合不变性约束公式**：

$$\text{Invariant}(A) = \forall t : \text{State}(A, t+1) \in \text{ValidStates}(A)$$

其中：
- $A$ = Aggregate
- $t$ = 时间点
- $\text{State}(A, t)$ = 聚合在时间 $t$ 的状态
- $\text{ValidStates}(A)$ = 聚合的所有有效状态集合

**聚合设计原则**：

| 原则 | 英文 | 说明 |
|------|------|------|
| 一致性边界 | Consistency Boundary | 聚合内部保证强一致性 |
| 通过根访问 | Access via Root | 外部只能通过聚合根访问内部对象 |
| 小聚合 | Small Aggregates | 聚合应该尽可能小 |
| 标识符引用 | Reference by ID | 聚合之间通过 ID 而非对象引用关联 |
| 最终一致性 | Eventual Consistency | 聚合之间使用最终一致性 |

### 3.3 Domain Service（领域服务）

当业务逻辑不属于任何实体或值对象时，使用 **Domain Service** 来封装。

```
┌────────────────────────────────────────────────────────┐
│          Domain Service: TransferService              │
├────────────────────────────────────────────────────────┤
│  + transfer(from: Account, to: Account, amount: Money) │
│    - 验证账户余额                                       │
│    - 执行转账逻辑                                       │
│    - 发布 Domain Event                                 │
└────────────────────────────────────────────────────────┘
```

**服务特征公式**：

$$\text{Service} = \{ o \mid \text{Stateless}(o) \land \neg \text{BelongsToEntity}(o) \land \neg \text{BelongsToValueObject}(o) \}$$

### 3.4 Repository（仓储）与 Factory（工厂）

#### Repository

**Repository** 封装数据持久化逻辑，提供类似集合的接口。

```
┌────────────────────────────────────────────────────────┐
│           Interface: Repository<T>                     │
├────────────────────────────────────────────────────────┤
│  + findById(id: TId): Optional<T>                      │
│  + save(entity: T): void                               │
│  + delete(entity: T): void                             │
│  + findBySpecification(spec: Specification<T>): List<T>│
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      Implementation: OrderRepositoryImpl               │
├────────────────────────────────────────────────────────┤
│  - dbContext: DatabaseContext                          │
│  - mapper: OrderMapper                                 │
├────────────────────────────────────────────────────────┤
│  + findById(id: OrderId): Optional<Order>              │
│  + save(order: Order): void                            │
│  ...                                                   │
└────────────────────────────────────────────────────────┘
```

**Repository 查询公式**：

$$\text{Query}(R, \text{spec}) = \{ e \in R \mid \text{spec}(e) = \text{true} \}$$

其中：
- $R$ = Repository
- $\text{spec}$ = 查询规约
- $e$ = 实体

#### Factory

**Factory** 封装复杂对象的创建逻辑。

```
┌────────────────────────────────────────────────────────┐
│               Factory: OrderFactory                    │
├────────────────────────────────────────────────────────┤
│  + createOrder(customerId: CustomerId): Order         │
│  + createOrderFromCart(cart: Cart): Order             │
│  + reconstituteFromEvents(events: List<Event>): Order │
└────────────────────────────────────────────────────────┘
```

---

## 四、Domain Events（领域事件）

**Domain Event** 表示领域中已经发生的、领域专家关心的事件。

### 4.1 事件分类

根据 Yan Cui 的分类：

| 类型 | 英文名 | 特点 |
|------|--------|------|
| 领域事件 | **Domain Events** | 限界上下文内部，轻量载荷，同步处理 |
| 集成事件 | **Integration Events** | 跨限界上下文，重量载荷，异步处理 |

### 4.2 事件结构

```
┌─────────────────────────────────────────────────────────────┐
│              Domain Event: OrderSubmitted                   │
├─────────────────────────────────────────────────────────────┤
│  - eventId: EventId                                         │
│  - eventType: "OrderSubmitted"                              │
│  - aggregateId: OrderId                                     │
│  - aggregateType: "Order"                                   │
│  - timestamp: DateTime                                      │
│  - payload: {                                               │
│      customerId: CustomerId,                                │
│      items: List<OrderItem>,                                │
│      totalAmount: Money                                     │
│    }                                                        │
│  - metadata: {                                              │
│      correlationId: String,                                 │
│      causationId: String                                    │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
```

**事件溯源公式**：

$$\text{State}(A, t) = \text{Apply}\left( \text{Initial}(A), \bigcup_{i=0}^{t} \text{Event}_i \right)$$

其中：
- $A$ = Aggregate
- $t$ = 当前版本
- $\text{Initial}(A)$ = 聚合的初始状态
- $\text{Event}_i$ = 第 $i$ 个事件
- $\text{Apply}$ = 应用事件到状态的函数

---

## 五、CQRS（Command Query Responsibility Segregation）

**CQRS** 是一种架构模式，将读操作与写操作分离。

### 5.1 CQRS 架构图

```
                    ┌─────────────────────────────┐
                    │        Client/UI            │
                    └─────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Command Side    │            │   Query Side     │
    │  (Write Model)   │            │  (Read Model)    │
    │                  │            │                  │
    │ ┌──────────────┐ │            │ ┌──────────────┐ │
    │ │ Command      │ │            │ │ Query        │ │
    │ │ Handler      │ │            │ │ Handler      │ │
    │ └──────────────┘ │            │ └──────────────┘ │
    │        │         │            │        │         │
    │        ▼         │            │        ▼         │
    │ ┌──────────────┐ │   Events   │ ┌──────────────┐ │
    │ │ Aggregate    │ │───────────▶│ │ Read Model   │ │
    │ │ Root         │ │            │ │ (Projection) │ │
    │ └──────────────┘ │            │ └──────────────┘ │
    │        │         │            │        │         │
    │        ▼         │            │        ▼         │
    │ ┌──────────────┐ │            │ ┌──────────────┐ │
    │ │ Event Store  │ │            │ │ Read DB      │ │
    │ │ (Write DB)   │ │            │ │ (Optimized)  │ │
    │ └──────────────┘ │            │ └──────────────┘ │
    └──────────────────┘            └──────────────────┘
```

### 5.2 CQRS 核心公式

**命令执行公式**：

$$\text{Execute}(C, A) = \begin{cases} \text{Events} & \text{if valid} \\ \text{Error} & \text{if invalid} \end{cases}$$

其中：
- $C$ = Command
- $A$ = Aggregate
- $\text{Events}$ = 产生的事件列表

**查询投影公式**：

$$\text{Project}(E, R) = \text{Update}(R, E)$$

其中：
- $E$ = Event
- $R$ = Read Model
- $\text{Update}$ = 更新读模型的函数

### 5.3 CQRS 与 DDD 的关系

| DDD 概念 | CQRS 对应 |
|----------|-----------|
| Aggregate Root | Command 处理单元 |
| Domain Event | 事件传播媒介 |
| Bounded Context | 可独立部署的 CQRS 服务 |

---

## 六、Event Sourcing（事件溯源）

**Event Sourcing** 是一种数据持久化模式，通过存储事件序列来记录状态变化，而非直接存储当前状态。

### 6.1 传统存储 vs 事件溯源

```
传统存储：
┌─────────────────────────────────────────────────────┐
│  Order Table                                        │
│  ┌─────────────────────────────────────────────────┐│
│  │ id | customer_id | status | total | created_at ││
│  │ 1  | 1001        | SUBMITTED | $500 | 2024-01-01││
│  └─────────────────────────────────────────────────┘│
│  只保存当前状态，丢失历史信息                         │
└─────────────────────────────────────────────────────┘

事件溯源：
┌─────────────────────────────────────────────────────┐
│  Event Store                                        │
│  ┌─────────────────────────────────────────────────┐│
│  │ seq | event_type       | payload    | timestamp ││
│  │ 1   | OrderCreated     | {...}      | 10:00     ││
│  │ 2   | ItemAdded        | {...}      │ 10:05     ││
│  │ 3   | ItemAdded        | {...}      │ 10:10     ││
│  │ 4   | OrderSubmitted   | {...}      │ 10:15     ││
│  └─────────────────────────────────────────────────┘│
│  保存完整历史，可重建任意时刻状态                      │
└─────────────────────────────────────────────────────┘
```

### 6.2 事件溯源公式

**状态重建公式**：

$$S_n = f(f(f(...f(S_0, E_1), E_2), ...), E_n) = \text{fold}(f, S_0, [E_1, E_2, ..., E_n])$$

其中：
- $S_n$ = 第 $n$ 版本的状态
- $S_0$ = 初始状态（通常为 null）
- $E_i$ = 第 $i$ 个事件
- $f$ = 应用事件的函数

**版本控制公式**：

$$\text{Version}(A) = \text{Count}(\text{Events}(A))$$

**乐观并发控制公式**：

$$\text{Save}(A, E_{n+1}, v_n) = \begin{cases} \text{Success} & \text{if } \text{CurrentVersion}(A) = v_n \\ \text{Conflict} & \text{otherwise} \end{cases}$$

---

## 七、Event Storming（事件风暴）

**Event Storming** 是一种协作式建模技术，由 Alberto Brandolini 提出，用于快速探索领域模型。

### 7.1 Event Storming 建模元素

| 颜色 | 元素类型 | 英文名 | 表示内容 |
|------|----------|--------|----------|
| 橙色 | 领域事件 | Domain Event | 已发生的事实（过去时） |
| 蓝色 | 命令 | Command | 触发事件的动作 |
| 黄色 | 聚合 | Aggregate | 处理命令、产生事件的实体 |
| 粉色 | 策略 | Policy | 事件触发的自动行为 |
| 绿色 | 读模型 | Read Model | 决策所需的数据视图 |
| 红色 | 热点 | Hot Spot | 问题、约束、痛点 |

### 7.2 Event Storming 工作流程图

```
┌────────────────────────────────────────────────────────────────────┐
│                      Event Storming Workshop                        │
│                                                                     │
│  Step 1: 探索领域事件 (橙色便利贴)                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ OrderCreated │  │OrderSubmitted│  │ OrderShipped │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│  Step 2: 识别命令 (蓝色便利贴)                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ CreateOrder  │  │ SubmitOrder  │  │ ShipOrder    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│  Step 3: 定义聚合 (黄色便利贴)                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │                    Order Aggregate                    │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                     │
│  Step 4: 识别策略、读模型、外部系统...                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## 八、Bounded Context 与 Microservices 的映射

### 8.1 映射关系类型

| 关系类型 | 说明 | 适用场景 |
|----------|------|----------|
| 1:1 | 一个 Bounded Context = 一个 Microservice | 理想状态，边界清晰 |
| 1:N | 一个 Bounded Context 分为多个 Microservice | 不同部分需要独立扩展 |
| N:1 | 多个 Bounded Context 合并为一个 Microservice | 简化运维，减少开销 |

### 8.2 映射决策公式

$$\text{Mapping} = \text{argmax}_{m \in M} \left( \text{Benefit}(m) - \text{Cost}(m) \right)$$

其中：
- $M$ = 所有可能的映射方案
- $\text{Benefit}(m)$ = 方案 $m$ 带来的收益（如独立部署、扩展性）
- $\text{Cost}(m)$ = 方案 $m$ 的成本（如运维复杂度、通信开销）

---

## 九、DDD 实践案例：电商订单系统

### 9.1 领域模型设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Bounded Context: Ordering                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Aggregate: Order                           │ │
│  │                                                               │ │
│  │  ┌─────────────────────────────────────────────────────────┐  │ │
│  │  │             Aggregate Root: Order                        │  │ │
│  │  │  - id: OrderId                                           │  │ │
│  │  │  - customerId: CustomerId                                │  │ │
│  │  │  - items: List<OrderItem>                                │  │ │
│  │  │  - status: OrderStatus [DRAFT, SUBMITTED, PAID, SHIPPED] │  │ │
│  │  │  - totalAmount: Money                                    │  │ │
│  │  │  - version: int (乐观锁)                                  │  │ │
│  │  ├─────────────────────────────────────────────────────────┤  │ │
│  │  │  + addItem(product, quantity): void                      │  │ │
│  │  │  + removeItem(itemId): void                              │  │ │
│  │  │  + submit(): OrderSubmitted Event                        │  │ │
│  │  │  + pay(paymentId): OrderPaid Event                       │  │ │
│  │  │  + cancel(): OrderCancelled Event                        │  │ │
│  │  └─────────────────────────────────────────────────────────┘  │ │
│  │                            │                                  │ │
│  │          ┌─────────────────┼─────────────────┐               │ │
│  │          ▼                 ▼                 ▼               │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │ │
│  │  │ OrderItem   │   │ Money       │   │ OrderStatus │         │ │
│  │  │ (Entity)    │   │ (ValueObj)  │   │ (Enum)      │         │ │
│  │  ├─────────────┤   ├─────────────┤   └─────────────┘         │ │
│  │  │- productId  │   │- amount     │                           │ │
│  │  │- quantity   │   │- currency   │                           │ │
│  │  │- unitPrice  │   └─────────────┘                           │ │
│  │  └─────────────┘                                             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Domain Services:                                              │ │
│  │  - OrderPricingService: calculateTotal(order) → Money         │ │
│  │  - OrderValidationService: validateForSubmission(order) → bool│ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Domain Events:                                                │ │
│  │  - OrderCreated, OrderItemAdded, OrderSubmitted,              │ │
│  │  - OrderPaid, OrderShipped, OrderCancelled                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 代码示例

```java
// Aggregate Root
public class Order {
    private OrderId id;
    private CustomerId customerId;
    private List<OrderItem> items;
    private OrderStatus status;
    private Money totalAmount;
    private int version;
    
    // Business Logic - 所有业务逻辑封装在聚合内部
    public void addItem(ProductId productId, int quantity, Money unitPrice) {
        if (status != OrderStatus.DRAFT) {
            throw new OrderCannotBeModifiedException(id);
        }
        
        // 检查是否已存在该商品
        Optional<OrderItem> existingItem = items.stream()
            .filter(item -> item.getProductId().equals(productId))
            .findFirst();
            
        if (existingItem.isPresent()) {
            existingItem.get().increaseQuantity(quantity);
        } else {
            OrderItem newItem = new OrderItem(
                OrderItemId.generate(),
                productId,
                quantity,
                unitPrice
            );
            items.add(newItem);
        }
        
        recalculateTotal();
    }
    
    public OrderSubmitted submit() {
        if (status != OrderStatus.DRAFT) {
            throw new OrderCannotBeSubmittedException(id);
        }
        
        if (items.isEmpty()) {
            throw new OrderCannotBeEmptyException(id);
        }
        
        this.status = OrderStatus.SUBMITTED;
        
        // 返回领域事件
        return new OrderSubmitted(
            id,
            customerId,
            totalAmount,
            LocalDateTime.now()
        );
    }
    
    private void recalculateTotal() {
        this.totalAmount = items.stream()
            .map(OrderItem::getSubtotal)
            .reduce(Money.ZERO, Money::add);
    }
}

// Value Object
public class Money implements ValueObject {
    private final BigDecimal amount;
    private final Currency currency;
    
    // 不可变：所有操作返回新实例
    public Money add(Money other) {
        if (!currency.equals(other.currency)) {
            throw new CurrencyMismatchException();
        }
        return new Money(amount.add(other.amount), currency);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Money)) return false;
        Money other = (Money) obj;
        return amount.equals(other.amount) 
            && currency.equals(other.currency);
    }
}

// Domain Event
public class OrderSubmitted implements DomainEvent {
    private final EventId eventId;
    private final OrderId aggregateId;
    private final CustomerId customerId;
    private final Money totalAmount;
    private final LocalDateTime occurredOn;
    
    // 事件是不可变的记录
}
```

---

## 十、DDD 的优势与挑战

### 10.1 优势

| 优势 | 说明 |
|------|------|
| 业务语言统一 | Ubiquitous Language 使技术与业务对齐 |
| 边界清晰 | Bounded Context 明确系统边界，降低耦合 |
| 可维护性 | 领域逻辑集中，易于理解和修改 |
| 可测试性 | 领域模型纯粹，单元测试容易 |
| 演进化 | 支持增量开发和持续重构 |

### 10.2 挑战

| 挑战 | 应对策略 |
|------|----------|
| 学习曲线陡峭 | 循序渐进，先实践 Tactical Design |
| 团队协作要求高 | Event Storming 促进沟通 |
| 过度设计风险 | 只在复杂领域应用 DDD |
| 技术与业务平衡 | 领域专家深度参与 |

### 10.3 适用场景判断公式

$$\text{DDDSuitability} = f(\text{Complexity}, \text{DomainKnowledge}, \text{TeamSkill}, \text{ProjectDuration})$$

当：
- $\text{Complexity} \geq \text{High}$
- $\text{DomainKnowledge} \neq \text{Trivial}$
- $\text{TeamSkill} \geq \text{Intermediate}$
- $\text{ProjectDuration} \geq \text{LongTerm}$

则 DDD 适用度高。

---

## 十一、DDD 与其他方法论的关系

```
                    ┌─────────────────────────────┐
                    │        Domain-Driven         │
                    │          Design             │
                    └─────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Model-Driven  │   │   Agile       │   │   SOA/        │
│ Engineering   │   │  Methodology  │   │ Microservices │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ DSL & Code    │   │ Iterative     │   │ Event         │
│ Generation    │   │ Refinement    │   │ Sourcing      │
└───────────────┘   └───────────────┘   └───────────────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │    CQRS       │
                                        └───────────────┘
```

---

## 十二、参考资源

### 经典书籍
1. **Domain-Driven Design: Tackling Complexity in the Heart of Software** - Eric Evans (2003)
   - 链接：https://www.domainlanguage.com/ddd/

2. **Implementing Domain-Driven Design** - Vaughn Vernon
   - 链接：https://www.amazon.com/Implementing-Domain-Driven-Design-Vaughn-Vernon/dp/0321834577

3. **Domain-Driven Design Distilled** - Vaughn Vernon
   - 链接：https://www.informit.com/store/domain-driven-design-distilled-9780134433421

### 在线资源
1. **DDD Community**：https://www.dddcommunity.org/
2. **Martin Fowler - DDD**：https://martinfowler.com/bliki/DomainDrivenDesign.html
3. **Microsoft DDD Guide**：https://docs.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/
4. **Event Storming**：https://www.eventstorming.com/
5. **Alberto Brandolini's Blog**：http://ziobrando.blogspot.com/

### 工具与框架
1. **Context Mapper**：https://contextmapper.org/
2. **Axon Framework**：https://axoniq.io/product-overview
3. **Lagom Framework**：https://www.lagomframework.com/

---

## 总结

Domain-Driven Design 的本质是**让软件设计回归到解决业务问题本身**。通过 **Ubiquitous Language** 消除沟通障碍，通过 **Bounded Context** 划分系统边界，通过 **Aggregate** 和 **Domain Event** 构建业务模型，DDD 为复杂系统的设计和演进提供了坚实的理论基础和实践指南。

在现代微服务架构、Event Sourcing、CQRS 等技术的加持下，DDD 正焕发新的生命力，成为企业级应用架构的重要方法论。