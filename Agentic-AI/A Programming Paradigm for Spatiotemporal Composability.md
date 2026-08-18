---
source_pdf: A Programming Paradigm for Spatiotemporal Composability.pdf
paper_sha256: 4d48478dc0b6222d9f74d7db10ee776449b1209eb112632336544d32a49db97f
processed_at: '2026-08-17T23:28:55-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲：这篇 paper 在干啥

Yo Andrej, 我换一种讲法，先讲故事再讲技术。

## 痛点：plugin 系统为什么这么烂

你装过 VSCode 插件吧？有没有发现一个很烦的事：**禁用一个插件经常要重启整个 VSCode**。为什么？

因为那个插件执行了 `activate()`，往 extension host 里塞了一堆东西：注册了 command、绑了快捷键、起了 background task、改了 UI 状态。VSCode 没办法把这些"副作用"单独撤掉，只能把整个 host 重启。

更尴尬的：VSCode 有个 `deactivate()` hook，但官方文档明确说它**只在 host 进程关闭时调用**，根本不是为了 live removal 设计的。而且 `activate` 和 `deactivate` 分离在代码两个地方，作者自己都容易写漏——漏了就 resource leak。

这不是 VSCode 独有的问题。OSGi、Eclipse、IntelliJ、所有 plugin 系统都差不多。Erlang 能 hot-swap 一个 module，但要靠人写 `code_change/3` 把老 state 迁到新格式。Webpack 的 HMR 要你在代码里手动声明 `module.hot.accept(...)`。

**根本困境**：现有的 cleanup 都靠**人手写**，靠**人记得 undo**。人不可靠，所以系统不敢真做细粒度卸载，退而求其次——重启整个 process。

## 这篇 paper 的主张

主张很简单：**让 runtime 自动追踪副作用，并自动撤销**，不需要人手写 cleanup。

怎么做到？核心 idea 是把 PL 课本里两个概念——**effect** 和 **coeffect**——从"编译时的 type annotation"升级成"运行时的 first-class 对象"，让 runtime 能直接操作它们。

- **Effect**：描述一个 computation 对环境做了什么修改（写状态、开连接、注册 handler）
- **Coeffect**：描述一个 computation 需要环境提供什么（依赖什么 service、读什么 config）

这两个东西在 Haskell / Koka 这种语言里是 type system 的一部分，用来静态推理。这篇 paper 的 move 是：**别在 type system 里折腾了，把 effect 和 coeffect reify 成 runtime data**，runtime 就能"看见"每个 component 干了什么、需要什么，然后自动管理。

## 时间维度：Revertible Effects

### 直觉

每干一件事，就同时记下"怎么撤掉这件事"。runtime 像记账一样把所有"撤销函数"累积起来，等要卸载时按相反顺序应用一遍。

类比：你装修房子，每装一件家具就贴个便签写着"怎么拆"。搬家时按便签反向操作，房子恢复毛坯。

### 形式化

把 context 记为 Γ。一个 effect 不是简单的 `Γ → Γ`（改 state），而是：

$$
e : \Gamma \to \Gamma \times (\Gamma \to \Gamma)
$$

变量解释：
- 输入 `γ : Γ`：当前 context
- 输出 `(δ, g)`：`δ` 是新 context，`g` 是这次操作的 inverse function
- `g` 满足 `g(δ) = γ`（在应用点处撤销自己）

为什么是 left inverse 而不是严格 inverse？因为撤销是单向的。malloc 给你指针 p，free(p) 把内存还回去——但 free 之后 heap layout 不一定跟没 malloc 时一样。我们只要求"在 malloc 之后那个状态，free 能回到等价状态"，不要求 free 之后能再 malloc 出同一个 p。

### Effect Context ∂Γ

为了让 runtime 能 track，引入一个"带账本"的 context：

$$
\partial\Gamma := \Gamma \times (\Gamma \to \Gamma)
$$

- 第一个 `Γ`：当前实际 state
- 第二个 `Γ → Γ`：**accumulator** φ，把所有 inverse 复合起来的"一键回滚"函数

初始状态 `(γ_0, id_Γ)`。每次 apply 一个 effect `e`：

$$
\text{track}(e)(\gamma, \varphi) = (\delta, \varphi \circ g) \quad \text{where } e(\gamma) = (\delta, g)
$$

意思：state 推进到 `δ`，accumulator 把新 inverse `g` 复合到末尾（注意是 `φ ∘ g`，因为 undo 要 LIFO）。

要回滚就 `recover(γ, φ) = (φ(γ), id_Γ)`——应用 accumulator，重置账本。

**Theorem 7（Soundness invariant）**：track 之后再 recover，等于直接 recover——effect 完全消化掉了。这就是 local temporal composability 的代数保证。

### Effect Iterator

实际中 effect 不是一次性的，是迭代的。比如一个 component activate 时可能：
1. 注册路由 `/foo`
2. 起一个 timer
3. 打开数据库连接
4. 订阅事件

每步都可能 fail，可能 await。所以 effect 升级成 iterator：

$$
\mathfrak{E}^{\text{iter}} := \mu\mathfrak{I}.\ \Gamma \to \Gamma \times (\Gamma \to \Gamma) \times \text{Maybe}(\mathfrak{I})
$$

每次 yield `(新state, inverse, 是否继续)`。这就是 reified delimited continuation，对应 JS 的 generator / `yield`。

Cordis 的 `ctx.effect(callback)` 就是这个的实现：callback 是个 generator，每个 `yield` 给一个 inverse，runtime 自动累积。callback 可以 `await`，可以 `while` 循环，完全自由。

### Independence — 跨 component 撤销的关键

到目前为止只保证 LIFO 撤销。但 plugin A 注册 `/foo`，plugin B 注册 `/bar`，现在卸载 A——B 还在，不能影响 B。

这要求 A 和 B 的 effect **互相独立**：A 的 transformation 跟 B 的 transformation commute。

形式化：两个 effect functions `e_1, e_2` 独立当：
1. 它们的 transformation monoid 互相 commute（forward map 和 inverse 都 commute）
2. 一个的 transformation 不改变另一个 yield 的 inverse

**Theorem 20**：如果 pairwise 独立，任何 permutation 顺序撤销都能回到初始状态。

但严格 commute 太强。怎么办？引入 **observational equivalence ≃**：两个 state 只要"任何 observer 都分不清"就算等价。比如两个 route 表，一个先加 `/foo` 再加 `/bar`，另一个反过来——查任何路由都一样，所以 `≃`-related，就算 commute。

**Theorem 42**：如果每个 effect 都通过 key 上的 operation 进行，且共享的 key 是 commutative 的（其 value 是一个"加/删独立"的表，如 route 注册），那么任意两个 effect function 都独立。

这就把 independence 从"强假设"变成"interface design obligation"：**provider 只要保证自己 key 的 operation commutative，consumer 就自动获得 independence**。

## 空间维度：Reactive Coeffects

### 直觉

每个 component 声明"我需要哪些 dependency"。runtime 监控 dependency 表的变化，一旦某个 component 的依赖全齐了就 activate 它，一旦某个依赖消失就 deactivate 它。

类比：你开了个餐厅，需要"食材供应商"+"厨师"+"收银系统"都到位才能营业。任何一个走了你就停业。

### 形式化

Coeffect context 是个 typed key-value 表：

$$
\Sigma := (k : K) \to \mathcal{V}_k
$$

- `K` 是 key 集合
- `ν : K → Type` 是 type family，每个 key `k` 关联 value type `ν_k`
- `σ : Σ` 是 partial function

`set(k, v)` 往表里塞值，`get(k)` 取值。**关键**：`set` 的类型恰好是 revertible effect——所以 coeffect 操作自动继承 temporal composability。

Component 声明它的 specification `d ⊆ K`（需要哪些 key）。Satisfaction predicate：

$$
\sigma \models d := \forall k \in d.\ k \in \text{dom}(\sigma)
$$

每次 σ 变化，对所有 component 做 classification：

$$
\text{notify}_d(\sigma, \sigma') = \begin{cases}
\text{activating} & \sigma \nvdash d \land \sigma' \models d \\
\text{deactivating} & \sigma \models d \land \sigma' \nvdash d \\
\text{neutral} & \text{otherwise}
\end{cases}
$$

Activating 触发 L-Begin（开始 reload），deactivating 触发 L-Leave（开始 unload）。

### Isolation 和 Interception

**Isolation**：同一个 key 在不同 context 解析到不同值。两层 mapping `K → R → V`，`R` 是 realm。比如测试环境用一个 mock database，生产环境用真的——同一个 key `database`，不同 realm。

**Interception**：给 dependency access 加 metadata。比如 filesystem 依赖可以带 metadata 声明"这个 component 只能读 `/tmp/` 下"。provider 在被调用时检查 metadata，决定是否放行。这是 **access control 的基础**。

## 把两个维度统一：Γ_∞

### Recursive Context Type

$$
\Gamma_\infty := \mu\Gamma.\ \Gamma \times (\Gamma \to \Gamma) \times \Sigma
$$

三个 projection：
- `Γ`：当前 state（recursive）
- `Γ → Γ`：accumulator
- `Σ`：coeffect table

直觉：把"state + 账本 + 依赖表"三元组递归地嵌套。effect 在这层 lift 自己（undo 一个 effect 本身也是 effect），所以 self-similar。

**任何 shared state 都 encode 成 key**——Σ 不只是 inter-component 依赖，subsumes 所有 shared mutable state。这是 paradigm 的 discipline：你要共享什么，就把它 bind 到一个 key。

### Component = (d, p, e)

$$
\mathfrak{C}_\Gamma := \mathfrak{D}_\Gamma \times \mathfrak{P}_\Gamma \times \mathfrak{E}_\Gamma^*
$$

- `d`：我需要什么（coefect specification）
- `p`：我提供什么（provision，我可能 install 的 keys）
- `e`：我做什么（witnessed effect function）

Fiber 是 component 的一次 instantiation，带 lifecycle state。

## Calculus：5 + 4 条规则

### Base Calculus（5 条）

- **O-Insert**：orchestrator 插入一个 fiber
- **O-Retire**：请求 retire（标 `τ=⊤`）
- **O-Remove**：真的删（要求 Inactive 且无 child）
- **L-Reload**：Inactive + 依赖满足 → 跑 effect function，记 accumulator
- **L-Unload**：Active + target 变了 → 跑 accumulator 回滚

Target view 比较的是 **provider 的身份**（fiber uid），不是 value。这样替换 provider 时（即使新旧值相等）也能触发 reload。

### 四个扩展

**Withdrawal（关键的 guard）**：L-Unload 加前提 `¬relied_n(γ)`——如果还有人通过 committed view 依赖我，我不能 unload。

流程是：先 L-Leave（标记 Unloading，停止提供 coeffect，但 inverse 还没跑）→ 依赖我的 fibers 看到 target 变 ⊥，自己也 L-Leave → 一层层往下 → 最底层无人依赖的 fiber 先 L-Unload → 释放 guard → 上一层 L-Unload → ... 

**Iteration**：把 L-Reload 拆成 L-Begin / L-Iter / L-Finish。每个 iteration boundary 可以 divert（如果 target 变了）。

**Asynchrony**：iteration 可以 `await`。一旦 launched，必须 land（inertia）——不能 abort in-flight 的 async operation，只能 land 之后再 unload。

**Failure**：iteration 可以 raise。raise 后进入 Unloading，应用已累积的 inverse，到达 Inactive(ξ)。**失败不传播到 parent**——一个 plugin 失败不影响兄弟 plugin。

## Metatheory：四个核心定理

### Preservation

Registry 的 tree 结构、provisions disjoint、committed view 合法性——每条规则都保持。

### Temporal Composability（Theorem 61）

**核心**：在一个 fiber 的 episode 内，apply 它的 accumulator 到当前 state，等价于"那些非它的步骤从 episode 开始 state 直接 apply 的结果"。

人话：**卸载一个 plugin，等于它从来没来过**。其他 plugin 的 effect 完全不受影响，不管中间怎么 interleave。

### Spatial Composability（Theorem 63）

**核心**：provider 必须先 activate，consumer 才能 activate；consumer 必须先 deactivate，provider 才能 withdraw。

而且 consumer 整个生命周期（包括它自己的 teardown）读到的 dependency 值是**稳定的**——provider 的 binding 在 consumer 没走之前不会变。

人话：**依赖关系自动协调，你不用手动排 load order**。

### Progress（Theorem 66）

系统不会死锁（总有 rule applicable），且会终止（每纤维的 step 数有上界 `(K+4)(V(n)+1)`，K 是 effect iterator 长度，V 是 target 变化次数）。

### Confluence（Theorem 73）

**最强结果**：任何动态历史（add / remove / replace / revert replacement）最终到达的 quiescent state，等同于"把最终 composition 一次性静态组装起来的结果"。

人话：**折腾一圈回到原点，state 跟你一开始就这么组装一模一样**。Dynamic history 不留痕迹。

这对 self-evolving agent 太关键了：agent 不停生成和替换自己的 components，但只要最终 active 的 component 集合一样，系统状态就一样——替换历史不影响最终结果。

## Cordis 实现

### `ctx.effect` — 唯一的 mutation primitive

```javascript
function effect(ctx, callback) {
  armed ← true
  task ← execute(callback, () ↦ armed)
  async function dispose() {
    if not armed then return
    armed ← false
    recover ← await task
    recover()
  }
  ctx.dispose ← dispose ∘ ctx.dispose  // 父级 composition
  return dispose
}
```

`armed` flag 同时 halt iteration + at-most-once recovery。`dispose ∘ ctx.dispose` 把 child 的 inverse 复合到 parent——卸载 parent 自动卸载 children。

### `ctx.set` — coeffect provision 也是 effect

```javascript
function set(ctx, key, value) {
  function callback() {
    realm ← ctx[@@isolate][key]
    ctx[@@store][realm] ← value
    notify(ctx, [key])  // 通知 dependents
    return function() {
      delete ctx[@@store][realm]
      notify(ctx, [key])
    }
  }
  return ctx.effect(callback)
}
```

装 binding 和撤 binding 都 `notify`——这就是 reactivity 的实现。

### Lifecycle state machine（Algorithm 5）

```
refresh(fiber):
  target ← compute_target(fiber)
  if target = fiber.target: return
  fiber.target ← target
  if fiber.inertia: return  // 已在 transition
  if target ≠ ⊥:
    reload(fiber)
  else:
    fiber.state ← UNLOADING  // L-Leave: 先标记
    unload(fiber)

unload(fiber):
  await all dependents reach INACTIVE  // guard
  await fiber.dispose()  // LIFO recovery
  if fiber.target = ⊥:
    fiber.state ← INACTIVE
  else:
    reload(fiber)  // chaining
```

三行对应 Theorem 63：
1. `fiber.committed ← resolve(...)` — commit view，整个生命周期读到的 binding 不变
2. `fiber.state ← UNLOADING` before any inverse — 让 dependents 提前看到 provider 离开
3. `await all dependents` — guard，等所有 consumer 走了才撤

### HMR 不需要 acceptance boundary

Webpack / Vite 的 HMR 要你写 `if (module.hot) module.hot.accept(...)`。Cordis 不需要——fiber 已经 bound 了 component 的所有 effects，dispose old fiber 自动 recover，新 fiber 自动 reinstall。三阶段：classify modules → detect stale entries → transactional reload（失败回滚）。

## Koishi：4000+ plugin 的生产验证

Koishi 是基于 Cordis 的聊天机器人框架，4 年积累 4000+ community plugins。IM adapters、database drivers、admin console、end-user features 都是 plugin。同一个 model 还跑在浏览器端（web console 是独立 Cordis app）。

验证三件事：
1. **Expressiveness**：一个完整 production 系统，host 框架只贡献 domain vocabulary
2. **Temporal composability without cognitive overhead**：plugin 作者不写 uninstall path，effect 自动 tracked + recovered
3. **Spatial composability across open ecosystem**：plugin 和 dependency 通常不同作者，只协调一个 coefect 连接

## 跟 AI agent harness 的关系

论文 Section 1.2.2 和 Section 8 都点名 self-evolving agent harness 是未来方向。一个 AI agent 持续生成和替换自己的 harness components：

- 替换一个 tool：旧 tool 的所有 side effect（注册的 handler、开的连接）自动撤掉，新 tool 干净 install
- 依赖拓扑频繁变：agent 生成的新 component 可能依赖旧 component 提供的 service，reactive coefect 自动 wire
- 失败隔离：一个生成的 component 有 bug，raise 后只影响它自己，agent 主循环继续跑
- Confluence：不管 agent 折腾多少轮，最终 state 只取决于"哪些 component 最终 active"

这正是 autonomous self-modification 需要的安全网——**形式化保证 recoverable + coordinated + continuous**。

## 一句话总结

**把 effect / coeffect 从编译时 type annotation 变成运行时 first-class 对象，runtime 就能自动追踪和撤销副作用，自动协调依赖关系——plugin 终于可以像 USB 设备一样热插拔了。**

参考链接：
- Cordis 源码: https://github.com/cordisjs/cordis
- Koishi: https://koishi.chat/
- Moggi 的 monadic effects: https://www.sciencedirect.com/science/article/pii/0890540191900524
- Plotkin & Power 的 algebraic effects: https://link.springer.com/chapter/10.1007/3-540-45315-6_24
- Petricek et al. 的 coeffects: https://dl.acm.org/doi/10.1007/978-3-642-39212-2_35
- Koka language: https://koka-lang.github.io/
- OSGi Declarative Services: https://docs.osgi.org/specification/osgi.cmpn/8.1.0/service.cm.html
- Erlang hot code loading: https://erlang.org/download/armstrong_thesis_2003.pdf
- React Hooks semantics: https://dl.acm.org/doi/10.1145/3763067
- Self-evolving agents survey: https://link.springer.com/article/10.1007/s11704-024-40231-1

---

# A Programming Paradigm for Spatiotemporal Composability — 深度技术讲解

Yo Andrej, 这篇paper很有意思, 它把 PL theory 里经典的 effect/coeffect 系统"reify"到 runtime, 用形式化的方式解决了一个工程界一直靠 hack 应付的问题: **怎么在不重启进程的前提下, 干净地装载/卸载软件组件**。下面我从 intuition 开始一层层剥开。

## 1. 核心问题:为什么现有方案都不够

论文一上来就指出 dynamic composition 有两个正交维度:

- **Temporal composability (时间维度)**: 卸载一个组件时, 它对共享环境做的所有修改必须能**完全可逆**。malloc 要 free, event listener 要 remove, route 注册要注销。
- **Spatial composability (空间维度)**: 组件之间要能**声明 + 发现 + reactive 解析**彼此的依赖, 当 provider 来去时 consumer 自动 activate/deactivate。

现有方案的局限:
- VSCode 的 extension host 不能单独卸载一个有代码的 extension (top 100 里 87 个含 code, 都要重启 host)
- VSCode 的 `extensionDependencies` 在 top 100 里只有 7 个真的用了, 因为 API 引导大家往 host-provided extension points 贡献, 而非互相依赖
- OS / container orchestrator 只能提供 coarse-grained 替代 (process / service 粒度), 代价是每次重启丢弃 in-memory state, 重建要几秒到几分钟

## 2. 关键洞察:Effect 和 Coeffect 是天然的形式化工具

这里有个非常漂亮的对应关系:

| Dynamic composability 维度 | 经典 PL 概念 | 方向 |
|---|---|---|
| Temporal | Effects | computation → environment |
| Spatial | Coeffects | environment → computation |

但 classical effect/coeffect systems 都是 **static instruments**: effect 在 lexical scope 内被 handler discharge, coeffect annotation 在执行前就 verify 完了。而 dynamic composition 要求这些保证在 runtime 持续成立, context 在不停演化。

**论文的 move**: 不要再往 type system 里加 annotation, 而是 **把 effect/coeffect 的概念结构 reify 成 first-class runtime entity**, 让 runtime 直接操作它们。

## 3. Revertible Effects — 时间维度的形式化

### 3.1 Effect Context ∂Γ

任何 impure function `f_impure : X → Y` 都可以纯化为 `f : Γ × X → Γ × Y`, 其中 Γ 是 context, 所有 side effect 都是 Γ 上的 transformation。

Effect 在 Γ 上构成一个 monoid (transformations under composition ∘), 满足 closure / associativity / identity。

要建模"可撤销"的 effect, 给每个 transformation `f` 配一个 left inverse `g` (即 `g ∘ f = id`, 不要求 `f ∘ g = id`)。注意是 **left inverse**, 因为 undo 是单向的: 我们只关心 `g(f(γ)) = γ`, 不关心 `f(g(δ)) = δ`。

**Definition 1 — Twisted composition**:
$$
(f_1, g_1) \circ (f_2, g_2) := (f_1 \circ f_2,\ g_2 \circ g_1)
$$
- 左操作数后作用, inverses 反向累积
- 这就是 monoid-of-transformations × 它的 opposite monoid, 记为 𝔗_Γ

**Definition 2 — Effect context**:
$$
\partial \Gamma := \Gamma \times (\Gamma \to \Gamma)
$$
状态 `(γ, φ)`:
- `γ : Γ` 是当前 context state
- `φ : Γ → Γ` 是 **accumulator**: 截至目前所有 effect 的 inverses 的复合, 即"恢复到初始状态的函数"

初始 effect context 是 `(γ_0, id_Γ)`。

### 3.2 track 与 recover

**Definition 3 — track**:
$$
\text{track}_\Gamma(f, g) : \partial\Gamma \to \partial\Gamma \\
\text{track}_\Gamma(f, g) = (\gamma, \varphi) \mapsto (f(\gamma),\ \varphi \circ g)
$$
意思是: 用 `f` 改 state, 把 `g` 复合到 accumulator 末尾 (注意 `φ ∘ g`, 因为 inverse 要 LIFO 应用)。

**Theorem 5**: track 是从 𝔗_Γ 到 ∂Γ → ∂Γ 的 monoid homomorphism。这意味着 **tracking 本身保持复合结构** — 这一点非常关键, 是后面 composability 的基石。

**Definition 6 — recover**:
$$
\text{recover}_\Gamma(\gamma, \varphi) = (\varphi(\gamma),\ \text{id}_\Gamma)
$$
把 accumulator 应用到当前 state, 把 accumulator 重置为 identity。

**Theorem 7 (Soundness invariant)**: 如果 `g(f(γ)) = γ`, 那么
$$
\text{recover}_\Gamma(\text{track}_\Gamma(f, g)(\gamma, \varphi)) = \text{recover}_\Gamma(\gamma, \varphi)
$$
也就是说, **track 后再 recover, 等于直接 recover** — effect 被完全消化掉了。

这就是 local temporal composability 的核心: 任何 track 过的 effect, 用 recover 都能拿回原状。

### 3.3 Effect Functions 𝔈_Γ 和 𝔈*_Γ

track 模型假设 inverse `g` 是 a priori 给定的, 一个 `g` 要 serve 所有 state。但实际中 inverse 是在 effect **应用时**才确定的 (比如 malloc 的 inverse 是 "free 这个具体的指针")。

**Definition 8**:
$$
\mathfrak{E}_\Gamma := \Gamma \to \Gamma \times (\Gamma \to \Gamma) \\
\mathfrak{E}_\Gamma^* := (e : \mathfrak{E}_\Gamma) \times \big((\gamma:\Gamma) \to ((\delta:\Gamma) \times (g:\Gamma\to\Gamma) \times ((\delta, g) = e(\gamma) \to g(\delta) = \gamma))\big)
$$

变量含义:
- `e(γ)` 返回 `(δ, g)`, 其中 `δ` 是新 context, `g` 是这次的 inverse
- 𝔈*_Γ 多了一个 witness: 对每个 `γ`, 如果 `e(γ) = (δ, g)`, 那么 `g(δ) = γ` 必须成立
- witness 把 inverse 约束在**它被应用的那个 state**, 不要求 `g` 在别处也对

**Effect composition ⋄ (Definition 9)**:
$$
(f \diamond g)(\gamma) = \text{let } (\delta, s) = g(\gamma) \text{ in} \\
\quad\quad \text{let } (\varepsilon, t) = f(\delta) \text{ in } (\varepsilon, s \circ t)
$$
- 先应用 `g` 得到 `(δ, s)`
- 再在 `δ` 上应用 `f` 得到 `(ε, t)`
- 总 inverse 是 `s ∘ t` (注意顺序: `g` 的 inverse `s` 在 outer, 因为 `g` 先执行, undo 时 `g` 的 undo 后执行)

**Theorem 10**: (𝔈_Γ, ⋄) 是 monoid, unit 是 `η_Γ(γ) = (γ, id_Γ)`。从 𝔗_Γ 到 𝔈_Γ 的映射 `(f, g) ↦ γ ↦ (f(γ), g)` 是 monoid homomorphism。

**effect transformation (Definition 12)**: 把 𝔈_Γ 上的 effect lift 到 ∂Γ 上:
$$
\text{effect}_\Gamma(e) : \partial\Gamma \to \partial^2\Gamma \\
\text{effect}_\Gamma(e) = (\gamma, \varphi) \mapsto \text{let } (\delta, g) = e(\gamma) \text{ in } ((\delta, \varphi \circ g),\ \text{track}_\Gamma(g, \text{pr}_1 \circ e))
$$

注意返回类型是 ∂²Γ — 这意味着 **undo 一个 effect 本身也是一个 effect**。`track_Γ(g, pr_1 ∘ e)` 是 undo 的 undo, 即把 effect 再做一次。这就是 recursive 的 ∂-tower, 后面 Γ_∞ 把它压平成单一 self-similar type。

### 3.4 Independence of Effects — 跨 component 的关键

到目前为止只保证 **LIFO 顺序撤销** (Theorem 16): 一个 component 自己的 effects 按 apply 的反向顺序 revert, 每个 inverse 拿到的就是它 apply 时产生的 state。

但动态组合需要的是: 一个 component 的 effect 可以**在另一个 component 的 effect 还在的时候**被撤销。比如 plugin A 注册了路由 `/foo`, plugin B 注册了路由 `/bar`, 现在卸载 A, B 的 `/bar` 还得保留。这是 **out-of-order reversion**。

**Definition 19 — Independence**: 两个 effect functions `e_1, e_2` 独立当:
1. 它们的 transformation monoids 互相 commute (Definition 17 给出 𝔐(e) 是 forward map + 所有 yielded inverses 生成的子 monoid)
2. 一个的 transformations 不扰动另一个 yield 的 inverse

**Theorem 20**: 如果 `e_1, ..., e_n` pairwise 独立, 从 `γ_0` 顺序 apply, 那么对任何 j, 可以在 `δ_n` 处单独 apply `g_j` 撤销 `e_j`, 而不影响其他 effects — 撤销后到达的状态等同于"从一开始就没 apply `e_j`"的状态。

**Corollary 21**: 任何 permutation 顺序 apply 所有 inverses 都能回到 `γ_0`。

**直觉**: independence 把"撤销顺序"从 LIFO 推广到任意顺序。LIFO 是免费的 (Theorem 16), 但 cross-component 需要 independence 假设。后面 Section 3.3.2 用 observational equivalence 来 supply 这个 independence。

## 4. Reactive Coeffects — 空间维度的形式化

### 4.1 Coeffect Context Σ

**Definition 22**:
$$
\Sigma := (k : K) \to \mathcal{V}_k
$$
- `K` 是 key 集合
- `ν : K → Type` 是 type family, 每个 key `k` 关联一个 value type `ν_k`
- `σ : Σ` 是有限 partial function, 把 `k ∈ dom(σ)` 映射到 `ν_k` 的 value

这是 **dependent partial function type**, 比 IoC 容器的 key-value map 强, 因为每个 key 有静态类型。

**get / set (Definition 23)**:
$$
\text{get}(k) : \Sigma \to \mathcal{V}_k \quad \text{(需要 } k \in \text{dom}(\sigma)\text{)} \\
\text{set}(k, v) : \Sigma \to \Sigma \times (\Sigma \to \Sigma) \quad \text{(需要 } k \notin \text{dom}(\sigma)\text{)} \\
\text{set}(k, v) = \sigma \mapsto (\sigma[k \mapsto v],\ \lambda\sigma'.\sigma' \setminus k)
$$

**关键观察**: `set(k, v)` 的类型恰好是 𝔈*_Σ — 它本身就是一个 **revertible effect**! 这就是 effect 和 coeffect 的 synergy: **coeffect 操作就是 effect, 自动继承 revertibility**。

### 4.2 Specification 和 Notification

**Definition 25 — Coeffect specification**: `𝔇_Σ := Set(K)`, 一个 component 声明它需要的 key 集合 `d ⊆ K`。

**Satisfaction predicate**:
$$
\sigma \models d := \forall k \in d.\ k \in \text{dom}(\sigma)
$$

**Definition 26 — notify classification**:
$$
\text{notify}_d(\sigma, \sigma') := \begin{cases}
\text{activating} & \text{if } \sigma \nvdash d \land \sigma' \models d \\
\text{deactivating} & \text{if } \sigma \models d \land \sigma' \nvdash d \\
\text{neutral} & \text{otherwise}
\end{cases}
$$

因为所有 σ 的 mutation 都通过 effect function (带 inverse), 每个 effect boundary 都能观测到 satisfaction 变化 — 这是 **reactivity 的代数基础**。

### 4.3 Isolation 和 Interception

**Isolation (Definition 28)**: 引入 realm table ρ, 同一个 key 在不同 context 解析到不同 binding:
$$
\Sigma^{\text{iso}} := (K \to R) \times ((r:R) \to \mathcal{V}_r)
$$
- `ρ : K → R` 把 key 映射到 realm identifier
- `σ : (r:R) → ν_r` 是 realm → value 的表

这本质是 **runtime ad-hoc polymorphism**: 同一个 logical key 可以在不同 context 解析到完全不同的值。set 仍是 effect (继承 revertibility), isolate 是 derived realization (不写共享表, 没有 inverse 要 track)。

**Interception (Definition 30)**: 给 dependency access 加 cross-cutting metadata:
$$
\Sigma^{\text{inter}} := ((k:K) \to \mathcal{M}_k) \times ((k:K) \to (\mathcal{M}_k \to \mathcal{V}_k))
$$
- 第一个是 context-carried metadata `ι`
- 第二个是 provider function `σ`, 接受 metadata 返回 value
- 每个 key 的 metadata 自带 monoid `(ℳ_k, ⊕_k, ε_k)`

访问时: `σ(k)(d(k) ⊕_k ι(k))` — component-declared metadata 和 context-carried metadata merge, 然后传给 provider。Right-biased 意味着 `ι(k)` 可以 override component 声明, 这是 **access control 的基础** (Section 6.3)。

## 5. Unified Context Γ_∞ — 编程范式的核心

### 5.1 Recursive Context Type

**Definition 32**:
$$
\Gamma_\infty := \mu\Gamma.\ \Gamma \times (\Gamma \to \Gamma) \times \Sigma
$$
三个 projection:
- `Γ`: 当前 context state (recursive)
- `Γ → Γ`: accumulator, 恢复这层的 effects
- `Σ`: coeffect context, 带 dependency 信息

**直觉**: 这是把 ∂-tower (Section 3.1 的 ∂Γ, ∂²Γ, ...) 和 Σ 用 μ-folding 压成单一 self-similar type。effect 在这层上 lift 自己 (effect maps 𝔈_{Γ_∞} 到自身)。任何要在 component 间共享的状态都可以 encode 为一个 key 的 dependency — Σ **subsumes 所有 shared mutable state**, 不只是 inter-component 依赖。

**Hierarchical composition**: parent context 聚合多个 child-level effects, 形成树状控制结构。"plug-in" 隐喻被字面实现:
- Loading = 执行 effects (plug in)
- Unloading = recover effects (unplug, 不影响其他 running components)
- 不同层级的 component 独立 loadable/unloadable

### 5.2 Observational Equivalence ≃

Theorem 7 的 equality 是 idealization — 物理状态不能完全恢复。free 释放内存块但不恢复 heap layout; generative name 不能被 discard 后 reuse。所以所有等式都要 read up to 一个 observational equivalence ≃。

**Definition 33**:
$$
\sigma \simeq \sigma' := \text{dom}(\sigma) = \text{dom}(\sigma') \land \forall k \in \text{dom}(\sigma).\ \sigma(k) \tilde{k} \sigma'(k) \\
\gamma \simeq \gamma' := \sigma_\gamma \simeq \sigma_{\gamma'}
$$
其中 `k̃_k` 是每个 key 自带的等价关系 (Definition 24)。

**Definition 34 — Indistinguishability**: 两个 value `v, v' : V` 不可区分 (`v ≈ v'`) 当且仅当对 key 上所有 operations 的所有 test (forward map 和 yielded inverse 的有限字), 在两者上都定义/不定义且产生相同 outcome。

**Lemma 35**: `≈` 是最粗的、operations respect 的等价关系 — 任何 admissible 的 `k̃_k` 都 contained in `≈_k`。

**Theorem 40**: distinct keys 上的 operations 天然 independent (因为它们读写不同的 key, Lemma 18(1) 把 generator-level commutation 扩展到整个 monoid)。

**Theorem 42**: 如果 `e_1, e_2` 是 coefect-mediated effect functions (Definition 41, 即由 key 上的 operations 串起来), 且两个 operations 共同出现的 key 都是 commutative 的, 那么 `e_1, e_2` 独立。

**这就是 independence 假设的来源**: 把所有 shared state 都 bind 到一个 key, 让 effect 都通过 key 上的 operation 进行, 那么 distinct keys 自动 independent, commutative keys 也 independent。**Commutativity 是 key 发布的 interface 的 property, 是 provider 的 obligation, 不是 consumer 的**。

例子: 一个 key 的 value 是 "route 表, 加 route / 删 route 独立" → commutative, 两个 plugin 注册路由顺序无关, 各自可独立撤销。一个 key 的 value 是 "ordered middleware chain" → 不 commutative, 中间件顺序敏感, 必须用 coefect 在 component 间 impose order。

## 6. Calculus of Dynamic Composition

### 6.1 Components 和 Fibers

**Definition 43 — Component**:
$$
\mathfrak{C}_\Gamma := \mathfrak{D}_\Gamma \times \mathfrak{P}_\Gamma \times \mathfrak{E}_\Gamma^*
$$
三元组 `(d, p, e)`:
- `d : 𝔇_Γ` — coefect specification (声明的依赖)
- `p : 𝔓_Γ := Set(K)` — provision (这个 component 可能提供的 keys)
- `e : 𝔈*_Γ` — witnessed effect function

**Definition 44 — Fiber**: component 的一次 instantiation, 带生命周期 state:
`⟨d, p, e, π, σ, τ, θ⟩`
- `π` — parent fiber (或 root marker)
- `σ : Σ` — fiber 自己的 coeffect table
- `τ : {⊥, ⊤}` — retirement flag
- `θ : Θ_Γ` — lifecycle state

**Definition 45 — Registry**: state γ 携带 `F_γ : 𝔑 → 𝔽_Γ`, 一个有限 partial function, parent pointers 形成以 root 为根的树。

**关键**: coeffect context **不是 stored 而是 derived**:
$$
\sigma_\gamma := \bigcup \{\sigma_m \mid m \in \text{dom}(F_\gamma),\ \theta_m = \text{Active}(-, -)\}
$$
即所有 **ACTIVE** fibers 的 table 的并集。Fiber 的 provisions 互不相交 (Definition 43), 所以每个 key 有 exactly one possible provider。**只有 coefect 部分这样记录**, 因为只有它是别的 fiber 能 declare against 的; 其他 effects 由 accumulator `g` 跟踪。

### 6.2 Base Calculus — 5 条规则

**Target view (Definition 46)**: 对每个 fiber n, 计算"它应该运行在哪个 resolution 上":
$$
\text{target}_n(\gamma) := \begin{cases}
\bot & \text{if } \tau_n \vee \neg(\gamma \models d_n) \\
(k \in d_n) \mapsto \text{provider}_k(\gamma) & \text{otherwise}
\end{cases}
$$

**Quiescence**: 每个 fiber 都达到 target view。

5 条规则:
- **O-Insert**: orchestrator 插入一个 fiber (要求 provisions 与现有 disjoint)
- **O-Retire**: 标记 `τ_n = ⊤` (请求 retire, 不是执行)
- **O-Remove**: 真的从 registry 删除 (要求 `θ_n = Inactive` 且无 child)
- **L-Reload**: `Inactive + target ≠ ⊥` → 应用 `e_n`, 记录 accumulator 和 committed view
- **L-Unload**: `Active + target ≠ ω` → 应用 accumulator `g`, 回到 Inactive

**Reactive discipline**: target view 变化就触发 transition, 不管是 retirement 还是 coefect 变化引起的。

### 6.3 Transitions in Progress — 4 个 extension

Base calculus 假设 transition 是 atomic / immediate / infallible。真实 runtime 都不是。Section 4.3 分四层放宽:

**Lifecycle state 扩展 (Definition 49)**:
$$
\Theta_\Gamma := \text{Inactive}(\zeta) \mid \text{Reloading}(i, g, \omega) \mid \text{Active}(g, \omega) \mid \text{Unloading}(g, \omega, \zeta)
$$
- `i` 是剩余的 effect iterator
- `g` 是已累积的 accumulator
- `ω` 是 committed view
- `ζ` 是 outcome (⊥ 或 error ξ)

**(1) Withdrawal (Section 4.3.1)** — 关键的 guard:

L-Unload 的前提加 `¬relied_n(γ)`, 其中:
$$
\text{relied}_n(\gamma) := \exists m \neq n, k \in d_m.\ \text{installed}_m(\gamma) \land \omega_m(k) = n
$$

意思是: **如果有别的 installed fiber 通过 committed view 依赖 n 提供的 key, n 不能 unload**。

L-Leave 把 fiber 标记为 Unloading 但**不立刻执行 accumulator**, 这让 fiber 停止提供 coeffect (它的 σ 离开 σ_γ 的 union), 依赖它的 fiber 的 target view 变成 ⊥, 触发它们自己的 deactivation。等所有 dependent 都 Inactive 后, guard 释放, L-Unload 才真正执行。

**为什么不会死锁**: 一旦 L-Leave 标记 n, `σ_γ` 不再包含 n 的 table, 任何 committed view 指向 n 的 dependent 的 target 立刻变 ⊥, 它们也开始 leave。Theorem 66 用 precedence relation `n ≺ m := p_n ∩ d_m ≠ ∅` 的 acyclicity 证明这点。

**(2) Iteration (Section 4.3.2)** — 把 atomic effect 换成 effect iterator:

**Definition 51**:
$$
\mathfrak{E}_\Gamma^{\text{iter}} := \mu\mathfrak{I}.\ \Gamma \to \Gamma \times (\Gamma \to \Gamma) \times \text{Maybe}(\mathfrak{I})
$$
每步返回 `(δ, g, o)`, `o` 是 `Nothing` (结束) 或 `Just(i')` (继续)。

**直觉**: 这就是 reified delimited continuation, 主流语言的 yield operator。每两个 iteration 之间有一个 boundary, 在那里可以 divert (如果 target view 变了)。

规则 L-Begin / L-Iter / L-Finish / L-Divert 把单一 L-Reload 拆成多步。

**(3) Asynchrony (Section 4.3.3)** — iteration 可能异步:

引入 opaque type constructor `Future`, 在 submission 和 resolution 之间外部状态可能变化。**Inertia**: 一旦 launched, iteration 必须 land, 不能 abort。所以 L-Divert 只能用 landing alternative (不能 abort 已 in-flight 的 iteration)。

**(4) Failure (Section 4.3.4)** — iteration 可能 raise:

**Definition 49 重读**:
$$
\mathfrak{E}_\Gamma^{\text{fail}} := \mu\mathfrak{I}.\ \Gamma \to \text{Either}(\Xi, \Gamma \times (\Gamma \to \Gamma) \times \text{Maybe}(\mathfrak{I}))
$$

**L-Raise**: `i(γ) = Left(ξ)` → 进入 `Unloading(g, ω, ξ)`, 应用已累积的 `g`, 到达 `Inactive(ξ)`。

**关键设计**: failure **不传播到 parent**, 只记录在 fiber 自己的 outcome 上。一个 component 失败不会 disable siblings — 这是 plugin host 想要的行为。

`Inactive(ξ)` 不能 re-enter L-Begin (premise 要求 `Inactive(⊥)`), 所以一个在它运行的环境下已经证明 unsound 的 effect function 不会被重试。

## 7. Metatheory — 四个核心定理

### 7.1 Preservation (Theorem 59)

Registry well-formedness 的四个 clause (Definition 58):
1. Parent pointer 落在 registry 内 (树形)
2. Distinct fibers 的 provisions disjoint
3. Installed fiber 的 committed view 是 total on `d_n` 且 valued in `dom(F_γ)`
4. 如果 installed `n` 且 `ω_m(k) = n` 那么 installed `m`

每条规则都 preserve 这四个 clause。**Guard on L-Unload 是 (3)(4) 的关键** — 它确保一个 fiber 不会在还有人依赖它的时候消失。

### 7.2 Temporal Composability (Theorem 61 — Recovery Exactness)

设步骤序列 pairwise independent, episode `[b, u]` of n, 令 `t_1 < ... < t_l` 是 `[b, u)` 中**不**作用在 n 上的步骤。那么:
$$
g_n^u(\gamma^u) \approx (\Psi^{t_l} \circ \dots \circ \Psi^{t_1})(\gamma^b)
$$

变量含义:
- `g_n^u` 是 n 在 u 时刻的 accumulator
- `γ^u` 是 u 时刻的 state
- `Ψ^t` 是步骤 t 的 state map (Table 1)
- `≈` 是 forget control fields 的等价 (只比较 tables 和 ambient state)

**直觉**: 在 n 的 episode 内, n 的 accumulator 应用到当前 state, 等价于"那些非 n 步骤从 episode 开始 state 直接 apply 的结果"。换句话说, **n 的 effects 完全被撤掉, 别的 fibers 的 effects 完全保留**, 不管中间怎么 interleave。

**Corollary 62 (Terminal recovery)**: episode 关闭时, `γ^{u+1}` ≈ 那些非 n 步骤从 `γ^b` 直接 apply 的结果 — 一个 fiber 卸载后, 它对 state 的贡献归零, 其他 fiber 不受影响。

### 7.3 Spatial Composability (Theorem 63 — Ordering)

两件事:
1. **L-Begin 仅在依赖满足时触发**: `step^t = L-Begin(m) ⇒ γ^t ⊨ d_m`
2. **Provider-consumer 顺序**: 如果 episode `[b', u']` of m 的 `ω_m(k) = n`, 那么:
   - `ω_m(k) = n` 在整个 episode 内不变 (committed view 是 fixpoint)
   - `b < b'` (provider 先 activate) 且 `u' < u` (consumer 先 deactivate)
   - `k ∈ dom(σ_n^t)` 且 `σ_n^t(k) = σ_n^{b'}(k)` (consumer 整个生命周期读到的值稳定)

**直觉**: 一个 consumer fiber 的整个生命周期, 包括它自己的 teardown, 都能稳定读到 provider 提供的 binding。Provider 的 withdrawal 在所有 consumer 都 Inactive 之后才发生。

**Theorem 64 (Resolution Coherence)**: Reloading 阶段每个 iteration 都 against 同一个 resolution `ω`。如果 target 在 iteration 之间变了, 要么 L-Finish 完成, 要么 L-Divert/L-Raise 退出并恢复。**一个 transition 不会跨越两个 resolutions**。

### 7.4 Progress (Theorem 66)

假设 `≺` acyclic, `len(e_n) ≤ K`, 名字集合 `N` 有限:
1. **No deadlock**: `¬quiet^t` 蕴含某个 lifecycle rule applicable
2. **Termination**: `S(n) ≤ (K+4)(V(n)+1)`, 其中 `V(n)` 是 target view 变化次数

证明思路: 沿 `≺` 归纳, target view 的每次变化要么消耗一个 `≺`-below fiber 的 step, 要么是该 fiber 自己 retirement 的 one-shot `τ_n` 写入。

### 7.5 Confluence (Theorem 73) — 最强的结果

**Support set (Definition 67)**: 一个 fiber 是 supported 当且仅当:
- 未 retired
- registering 它的 fiber 也 supported (或它是 root)
- 它 declare 的每个 key 都有 supported fiber 提供

**Lemma 68**: `≺` acyclic 时, support 关系 `⊲` well-founded, A 是 τ/π/d/p 的函数 (与 schedule 无关)。

**Lemma 70**: 在 quiescent state, 没有失败 fiber, 所有 component total on provision 时, **support set = ACTIVE fibers 集合**。

**Theorem 73 (Confluence)**:
1. **Canonical form**: 任何到达 quiescent `γ^T` 的序列, 都能 reduce 成 canonical 序列: 同样的 orchestration steps (按原序), 然后按 ⊲ 的一个 linearization, 每个 supported fiber 一个 episode
2. **Confluence**: 两个这样的 canonical 序列, 在 renaming (Lemma 56) 后, 到达 ≃-equal 且 ≈-equal 的状态

**直觉**: **dynamic history 不留痕迹**。不管你怎么 add/remove/replace provider/revert replacement, 最终 quiescent state 等同于"把最终 composition 一次性静态组装起来的结果"。这是 dynamic composition 的 consistency-with-from-scratch 性质, 类比 incremental computation 的 change propagation。

**Failure 例外**: failure 是真正的 divergence 源, 因为 raise 依赖 state — 不同 schedule 可能在不同地方 fail。但 Corollary 62 保证 failed fiber 对 state 的贡献为零, 所以两个 quiescent state 只在 failed fiber 的 lifecycle state 上不同。

## 8. Implementation — Cordis

### 8.1 Core Library

**ctx.effect (Algorithm 1)**: 唯一的 mutation primitive。coefect provision、component instantiation、所有 context-mutating 操作都 reduce 到 `ctx.effect` 调用。

```javascript
async function execute(callback, guard) {
  iter ← callback()
  inverse ← id
  while guard() {
    (value, done) ← await iter.next()
    if value then inverse ← value ∘ inverse  // LIFO
    if done then break
  }
  return inverse
}

function effect(ctx, callback) {
  armed ← true
  task ← execute(callback, () ↦ armed)
  async function dispose() {
    if not armed then return
    armed ← false  // 同时 halt in-flight + at-most-once
    recover ← await task
    recover()
  }
  ctx.dispose ← dispose ∘ ctx.dispose  // 父级 composition
  return dispose
}
```

**关键设计**:
- `armed` flag 同时 halt iteration + at-most-once recovery (避免 inverse 在错误 state 应用)
- `dispose ∘ ctx.dispose` 把 child 的 inverse 复合到 parent — 这是 ∂²Γ 的递归结构
- guard 在 component level 换成 `fiber.target = target₀` 的稳定性检查 (对应 L-Divert)

### 8.2 Coeffect Operations

三个 symbol-keyed slot:
- `@@store`: value store σ (realm → value)
- `@@isolate`: realm table ρ (key → realm)
- `@@intercept`: interception table ι (key → metadata)

**ctx.set (Algorithm 2)**: 实现 `set(k, v)`, 通过 `ctx.effect` 调用, 自动 tracked + recoverable。安装和移除都调用 `notify` 通知 dependents。

**notify (Algorithm 3)**: 遍历所有 live fibers, 检查 changed key 是否在它的 `inject` 中且 realm 匹配, 是则 `refresh(fiber)`。这就是 Definition 26 的 reactive classification 在 runtime 的实现。

### 8.3 Component Lifecycle (Algorithm 5)

```
refresh(fiber):
  target ← compute from coeffect store
  if target = fiber.target then return
  fiber.target ← target
  if fiber.inertia then return  // 已在 transition 中
  if target ≠ ⊥ then
    fiber.state ← LOADING
    fiber.inertia ← reload(fiber)
  else
    fiber.state ← UNLOADING  // L-Leave: 先标记, 让 dependents 看到
    fiber.inertia ← unload(fiber)

reload(fiber):
  target₀ ← fiber.target
  fiber.committed ← resolve(fiber.inject)  // commit view
  recover ← await execute(fiber.apply, () ↦ fiber.target = target₀)
  fiber.dispose ← recover ∘ fiber.dispose
  if fiber.target = target₀ then
    fiber.state ← ACTIVE
    notify(fiber.ctx, provided(fiber))
  else
    fiber.state ← UNLOADING  // chaining
    fiber.inertia ← unload(fiber)

unload(fiber):
  await all(notify(...).map(f ↦ f.await()))  // guard: drain dependents
  await fiber.dispose()  // LIFO recovery
  fiber.dispose ← id
  fiber.committed ← ⊥
  if fiber.target = ⊥ then
    fiber.state ← INACTIVE
  else
    fiber.state ← LOADING  // chaining back to reload
    fiber.inertia ← reload(fiber)
```

**三个关键 line 对应 Theorem 63**:
1. `fiber.committed ← resolve(...)` (Line 14) — commit view, unload 时才 discard, 保证 consumer 整个生命周期读到同样的 binding
2. `fiber.state ← UNLOADING` (Line 10) — L-Leave, 在任何 inverse scheduled 之前, 让 dependents 看到 provider 已离开
3. `await all(notify(...))` (Line 25) — guard on L-Unload, 等待所有 dependent 到达 INACTIVE

### 8.4 Context Access via Proxy (Algorithm 6)

```
resolve(ctx, key):
  fiber ← ctx.fiber
  repeat:
    if key ∈ fiber.committed then return fiber.committed[key]
    if key ∈ fiber.inject then throw INACTIVE_ACCESS
    if fiber = root then throw UNDECLARED_ACCESS
    fiber ← fiber.parent.fiber
```

**关键区别**: `ctx.get(key)` 是裸 lookup (返回 value 或 nothing, 不 fail); `ctx[key]` (proxy) 是 **against the accessing fiber's own view**, 在 use point 强制 coefect specification `d`。这是 capability-based access control: 一个 component 只能 access 它 declare 过的 dependency。

## 9. Component Loader — 声明式配置 + HMR

### 9.1 Declarative Configuration

每个 entry 有: `id, url, isolate, intercept, config, disabled`。这六个字段恰好对应 support set (Definition 67) 读取的 `τ, π, d, p` 加上 runtime state。

**Reconciliation 的合理性**来自 metatheory:
- Theorem 73: quiescent state 是 final config 的函数, 中间过程不影响最终结果
- Theorem 66: 系统会 quiesce
- Corollary 62: 重建一个 entry 撤回它的贡献, 不影响周围 fibers
- Theorem 63: 不需要 orchestrator 安排 load order, dependency 自动等待

**Managed realms (Algorithm 7)**: entry 可能在 group 间移动, realm 重新分配。用 delimiter `δ_k` (一个 symbol, 每个 key 一个) 在 context 上写 tag, inherited by descendants。Test: `γ'[δ_k] = d₁` 当且仅当 `γ'` 是从 entry 的 context derived 的。这精确判断"binding 是不是 entry 自己的", 决定要不要跟着 move。

### 9.2 Hot Module Replacement (Algorithm 8-10)

**三阶段**:
1. **Module classification**: 用 import graph 的 fixed-point, accepted (可热替换) / declined (必须重启)
2. **Stale-entry detection**: walk 每个 entry 的 dependency tree, 跟 accepted 取交集
3. **Transactional reload**: invalidate caches → 重新 import → swap fiber。失败则 restore caches + 从 backup 重建, 保证不会 half-reloaded

**关键**: HMR 不需要 developer-annotated acceptance boundaries (对比 webpack / vite), 因为 fiber 已经 bound 了 component 的所有 effects 和 coeffects — dispose old fiber 自动 recover, 新 fiber 自动 reinstall。

## 10. Koishi Case Study

Koishi 是基于 Cordis 的聊天机器人框架, 4 年 4000+ community plugins:
- IM adapters, database drivers, admin consoles, end-user features
- 同一 model 还用于 web console (浏览器端独立 Cordis app)

**验证的点**:
1. **Expressiveness + generality**: 一个 production 系统, host 框架只贡献 domain vocabulary
2. **Temporal composability without cognitive overhead**: plugin 作者不需要写 uninstall path, 通过 context 的 effect 自动 tracked + recovered
3. **Spatial composability across open ecosystem**: plugin 和 dependency 通常不同作者, 只协调一个 coefect 连接它们

## 11. 我的几点观察

### 11.1 设计哲学的漂亮之处

这篇 paper 最 elegant 的地方是 **把 PL theory 的 static 概念 reify 到 runtime**。effect/coeffect 在传统 PL 里是 type-level annotation, 用来静态推理。这里把它们做成 first-class runtime entity, 让 runtime 能直接操作 inverses 和 specifications。这等于把"编译时验证的东西"变成"运行时强制保证的不变量"。

### 11.2 Twisted Composition 的意义

Definition 1 的 twisted composition `(f₁, g₁) ∘ (f₂, g₂) = (f₁∘f₂, g₂∘g₁)` 看起来怪, 但其实非常自然: **forward 顺序执行, inverse 反向执行**。这就是 stack-based 的 LIFO, 但被抽象成 monoid 结构, 让 track 成为一个 monoid homomorphism (Theorem 5)。这个 homomorphism 性质是 composability 的代数基石 — 它保证 "tracking + composition 可交换", 也就是说"先 compose 再 track" = "先 track 再 compose"。

### 11.3 Observational Equivalence 的妙用

Definition 33 的 `≃` 看起来只是技术细节, 但其实是 independence 假设的"补完"。Theorem 16 只给 LIFO reversion, 跨 component 需要 independence (Theorem 20)。但严格 independence 太强 (要求 `f ∘ g = g ∘ f` 严格相等)。引入 `≃` 后, "两个 operation 留下 `k̃_k`-related 的值" 也算 commute, 这让 commutative key 的范围大大扩大 — 比如 route 表的"加 /foo"和"加 /bar"严格来说让表处于不同状态, 但 observational equivalent (查任何路由都一样), 所以 commute。

### 11.4 跟 React Hooks 的对比

论文 Section 7.3 提到 React 的 `useEffect`: 它返回 cleanup, runtime 在 unmount 或 re-exec 前调用。但 `useEffect` 不能在 conditional / loop / nested function 里调用, effect body 不能是 async 或 iterator — 所以 effects 不能从其他 effects 组装, 也没有 composite inverse 可推导。Cordis 的 effect 是普通 operation, 自由 compose, 可以 async, 只要求 atomic effect 手写 inverse。

这个对比其实点出一个深层问题: **React Hooks 的"rules of hooks"是为了在缺乏 first-class effect tracking 的语言里模拟 effect system**。Cordis 通过 reify context 把这个限制消除了。

### 11.5 跟 OSGi 的对比

OSGi Declarative Services / iPOJO 是最接近的 precedent: 声明 provided/required services, runtime 自动 activate/deactivate。但两个限制:
1. Deactivation callback 是手写的, 资源安全靠开发者纪律
2. Callback 是 synchronous 的, 没法 await 异步 teardown

Cordis 的 inertial Unloading state (Section 4.3.3) 解决了第二点: 异步 teardown 完成后才响应进一步变化。

### 11.6 跟 Algebraic Effects 的对比

Effekt 语言 (Brachthäuser et al.) 把 effect types 重新解释为 capabilities — effect type 表达"computation 从 context 需要什么", 跟 Cordis 把 context 当 mediator of capabilities 的视角很像。但两个区别:
1. **目的**: algebraic effects 让 effects 可见以 enable modular interpretation (一个 operation 多个 handler 语义); Cordis 让 effects 可见以 enable tracking + reversion
2. **设置**: Effekt 在 type level 静态 discipline, capability 默认 second-class + lexical scope; Cordis 在 runtime discipline, 目标是 complete resource recovery on removal

### 11.7 Confluence 的含义

Theorem 73 是最强的结果: **dynamic history 不留痕迹**。这意味着你可以把 Cordis application 当作 statically assembled 来 reason about。一个 orchestrator add/remove/replace/revert, 最终到达的状态等同于"一次性写下最终 composition"。

这跟 self-evolving agent harness (Section 1.2.2) 的相关性: 如果一个 agent 不停生成和替换自己的 components, confluence 保证最终状态只取决于"哪些 components 最终是 active 的", 不取决于替换历史。这是 **recoverable + coordinated + continuous self-evolution** 的形式化基础。

### 11.8 局限性

论文也诚实指出:
- **Finiteness of N** 是假设, 不是推导 — 一个 component 不能无限制 register 自己的 instances
- **`≺` acyclic** 是假设 — 互相依赖的 components 永远 inactive (但这是 predictable 的, runtime 可报告)
- **Totality on provision** 是 confluence 的条件 — 一个 component 必须在 finish 时 install 所有 declared key
- **System boundary** (Section 6.1): 外部 emissions (write 到 file, send 到 network) 无法严格 recover, 只能 withhold 或 compensate
- **Single ecosystem, single language**: Koishi 是 TypeScript, 没法分离 paradigm 的贡献和语言的贡献

### 11.9 跟你 (Karpathy) 关心的 agent harness 的连接

论文 Section 1.2.2 和 Section 8 都强调 self-evolving agent harness 是未来 validation 方向。一个 AI agent 持续生成和替换自己的 harness components, 需要的正是:
- Temporal composability: 替换组件时完全 recover 旧组件的 effects, 不积累垃圾
- Spatial composability: 依赖拓扑频繁变化时, 自动 coordinate, 不引入循环依赖

Cordis 的 metatheory (preservation / progress / confluence) 提供了形式化保证: 替换历史不影响最终状态, 系统总能 quiesce, 失败的 component 不影响 siblings。这正是 autonomous self-modification 需要的安全网。

## 12. 参考链接

- **论文本身**: 这次 attach 的文件, arXiv 版本应该会出, 关注 Yifan Shi / Wei Zhang / Tianyi Cui (PKU + DeepSeek-AI)
- **Cordis 实现**: https://github.com/cordisjs/cordis (核心库)
- **Koishi (案例研究)**: https://koishi.chat/
- **Effect 系统 (Moggi)**: https://www.sciencedirect.com/science/article/pii/0890540191900524
- **Algebraic Effects (Plotkin & Power)**: https://link.springer.com/chapter/10.1007/3-540-45315-6_24
- **Effect Handlers (Plotkin & Pretnar)**: https://link.springer.com/chapter/10.1007/978-3-642-00590-9_7
- **Coeffects (Petricek et al.)**: https://dl.acm.org/doi/10.1007/978-3-642-39212-2_35
- **Graded effects + coeffects (Gaboardi et al.)**: https://dl.acm.org/doi/10.1145/2951913.2951939
- **Koka language**: https://koka-lang.github.io/
- **OCaml 5 effects**: https://dl.acm.org/doi/10.1145/3408995
- **OSGi Declarative Services**: https://docs.osgi.org/specification/osgi.cmpn/8.1.0/service.cm.html
- **iPOJO / Gravity**: https://dl.acm.org/doi/10.1109/ICSE.2004.1317483
- **Erlang/OTP hot code loading**: https://erlang.org/download/armstrong_thesis_2003.pdf
- **Kitsune (DSU for C)**: https://dl.acm.org/doi/10.1145/2629460
- **R-OSGi (distributed)**: https://link.springer.com/chapter/10.1007/978-3-540-76778-7_1
- **Software Transactional Memory**: https://dl.acm.org/doi/10.1145/224964.224987
- **Reversible Computing (Janus)**: https://dl.acm.org/doi/10.1145/1244381.1244404
- **RCCS (reversible process calculus)**: https://link.springer.com/chapter/10.1007/978-3-540-28644-8_19
- **Nooks (kernel extension recovery)**: https://dl.acm.org/doi/10.1145/945445.945466
- **Shadow Drivers**: https://dl.acm.org/doi/10.1145/1189256.1189257
- **React-tRace (React Hooks semantics)**: https://dl.acm.org/doi/10.1145/3763067
- **Context-oriented Programming**: https://www.jot.fm/contents/issue_2008_03/article4.html
- **Aspect-Oriented Programming**: https://link.springer.com/chapter/10.1007/BFb0053381
- **Saga pattern**: https://dl.acm.org/doi/10.1145/38713.38742
- **Self-evolving agents survey**: https://link.springer.com/article/10.1007/s11704-024-40231-1
- **MemGPT (LLM as OS)**: https://arxiv.org/abs/2310.08560
- **Tool Learning with Foundation Models**: https://dl.acm.org/doi/10.1145/3704435

---

总结一句: 这篇 paper 用 PL theory 的语言 (effect / coeffect / monoid / observational equivalence) 给 dynamic composition 一个形式化基础, 同时在工程上实现成 Cordis 并用 Koishi (4000+ plugins) 验证。对 agent harness 这种需要持续 self-modification 的场景, 它的 confluence + recovery exactness 提供了"替换历史不留痕迹"的强保证, 这正是 autonomous self-evolution 缺失的安全网。
