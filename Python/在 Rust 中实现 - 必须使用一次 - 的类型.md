# 文章概览：在 Rust 中实现"必须使用一次"的类型

这篇文章探讨了一个非常有趣且有点"邪恶"的 Rust 编程技巧：**如何创建一个必须被恰好使用一次的类型** `UseOnce<T>`。

---

## 核心概念：线性类型

**线性类型** 来自线性逻辑，其核心约束是：**每个值必须被使用恰好一次**——不能不用，也不能用多次。

Rust 的 affine type system（仿射类型系统）只保证"最多使用一次"（所有权转移后原变量不可用），但不保证"必须使用一次"（你可以忘记使用它）。

---

## 实现原理解析

### 核心数据结构

```rust
pub struct UseOnce<T>(MaybeUninit<T>);
```

**为什么用 `MaybeUninit<T>` 而不是 `Option<T>`？**
- `Option<T>` 需要额外的空间存储 discriminant（判别标志位）
- `MaybeUninit<T>` **零空间开销**，只是告诉编译器"这块内存可能未初始化"

### 关键方法：`consume`

```rust
pub fn consume<F, R>(self, f: F) -> R
where
    F: FnOnce(T) -> R,
{
    let mut this = ManuallyDrop::new(self);      // ①
    let mut val = MaybeUninit::uninit();          // ②
    std::mem::swap(&mut this.0, &mut val);        // 交换内部值
    unsafe {
        let val = val.assume_init();              // 安全：此时val已包含真实值
        f(val)
    }
}
```

**① `ManuallyDrop` 的作用：**
- 包装 `self`，使其**不会自动调用 Drop**
- 编译器会推理：如果调用了 `consume`，值已被移出，就不需要执行 Drop

**② 交换技巧的妙处：**
```
交换前：
  this.0 = MaybeUninit::new(actual_value)
  val    = MaybeUninit::uninit()  (垃圾值)

交换后：
  this.0 = MaybeUninit::uninit()  (垃圾值)
  val    = MaybeUninit::new(actual_value)
```

这样 `this`（即原 `self`）被丢弃时，里面的值已经是"空壳"，不会 double-free。

---

## 黑魔法：编译时 panic

```rust
impl<T> Drop for UseOnce<T> {
    fn drop(&mut self) {
        const {
            panic!("UseOnce instance must be consumed!")
        }
    }
}
```

### 这是整篇文章最 clever 的地方！

**`const { panic!(...) }` 是什么？**

这是 Rust 的 **const block** 或 **inline const** 特性，它在**编译时**评估表达式。当编译器尝试实例化这个 `drop` 方法时，会遇到一个必然 panic 的常量，从而**在编译期报错**！

**工作流程：**

```
情况1: 没有调用 consume
┌─────────────────────────────────────┐
│ UseOnce 实例离开作用域              │
         ↓
│ 编译器生成 drop 代码                │
         ↓
│ 评估 const { panic!(...) }          │
         ↓
│ ❌ 编译错误！                        │
└─────────────────────────────────────┘

情况2: 调用了 consume
┌─────────────────────────────────────┐
│ consume 被 call                      │
         ↓
│ ManuallyDrop 包装 self              │
         ↓
│ 值被 swap 出来并消费掉               │
         ↓
│ ManuallyDrop 被丢弃，不调用 drop    │
         ↓
│ ✅ 编译通过！                        │
└─────────────────────────────────────┘
```

---

## 严重缺陷与"被诅咒"的真相

### 缺陷1：`mem::forget` 可以绕过

```rust
let instance = UseOnce::new(42);
std::mem::forget(instance);  // 完全绕过检查，没有任何错误！
```

`mem::forget` 会"忘记"值的存在，不调用 Drop，也就不会触发编译错误。

### 缺陷2：身份函数泄露内部值

```rust
let instance = UseOnce::new(42);
let leaked = instance.consume(|v| v);  // 直接把 v 返回了！
```

作者提供了改进方案，只暴露引用：

```rust
pub fn consume<F, R>(self, f: F) -> R
where
    F: FnOnce(&T) -> R,  // 注意这里是 &T，不是 T
```

### 缺陷3：编译器优化依赖——最致命的问题！

```rust
fn foo() {}

fn main() {
    let instance = UseOnce::new(41);
    foo();  // ← 插入一个函数调用
    let _result = instance.consume(|v| v + 1);
}
```

**这段代码竟然编译失败！** 为什么？

#### 第一性原理分析：

编译器在没有优化的情况下，会做保守的分析：

1. `foo()` 是一个外部函数调用
2. 编译器不知道 `foo()` 是否会 panic
3. 如果 `foo()` panic，那么 `instance` 就不会被 consume
4. 因此，编译器**必须**生成 `instance` 的 drop 代码
5. 于是触发 `const panic!`

**关键依赖：** 这个技巧依赖编译器能够**静态推断**某个 Drop 实现永远不会被执行。如果编译器无法确定这一点（比如中间有函数调用），就会生成 drop 代码并报错。

---

## 更深的黑暗魔法：`prevent_drop` 的方法

文章提到了 [prevent_drop crate](https://crates.io/crates/prevent_drop) 的思路：

```rust
impl<T> Drop for UseOnce<T> {
    fn drop(&mut self) {
        extern "C" {
            fn __prevent_drop_link_error();
        }
        unsafe { __prevent_drop_link_error(); }
    }
}
```

**这个函数 `__prevent_drop_link_error` 根本不存在！**

- 如果 drop 被生成，**链接阶段**会报错
- 错误被推迟到链接时，但至少能工作

**代价：** 错误信息更加晦涩，且需要确保符号名不会意外被链接。

---

## 技术细节深入

### `MaybeUninit` 的内存布局

```
┌────────────────────────────────────────────────┐
│ MaybeUninit<T> 的内存表示                       │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────────────────────────────────┐  │
│  │  T 类型的内存空间（可能包含垃圾数据）      │  │
│  │  大小 = size_of::<T>()                    │  │
│  │  对齐 = align_of::<T>()                   │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  编译器假设：这块内存可能是未初始化的           │
│  你必须用 unsafe 代码保证正确使用               │
└────────────────────────────────────────────────┘

对比 Option<T>:

┌────────────────────────────────────────────────┐
│ Option<T> 的内存表示                           │
├────────────────────────────────────────────────┤
│  ┌─────────────┬────────────────────────────┐  │
│  │ discriminant │      T 的空间              │  │
│  │   (1 byte+)  │                           │  │
│  └─────────────┴────────────────────────────┘  │
│  额外开销：至少 1 byte（通常有 padding）        │
└────────────────────────────────────────────────┘
```

### `Pin` 在 `consume_mut` 中的作用

```rust
pub fn consume_mut<F, R>(self, f: F) -> R
where
    F: FnOnce(Pin<&mut T>) -> R,
{
    // ...
    let pinned = Pin::new_unchecked(&mut val);
    f(pinned)
}
```

**`Pin<&mut T>` 的语义：**

- 普通的 `&mut T` 允许你把 `T` move 出来（比如 `std::mem::replace`）
- `Pin<&mut T>` 承诺：**这个引用指向的值不会被移动**
- 这防止了类似 `*pinned = new_value` 的"偷梁换柱"

---

## 为什么说这是 "Cursed"（被诅咒的）？

| 问题 | 说明 |
|------|------|
| **依赖未指定的编译器行为** | 编译器如何决定是否生成 drop 代码，并没有明确规范 |
| **错误信息晦涩** | 用户看到的是 const evaluation 失败，而不是友好的类型错误 |
| **不真正的线性类型** | `mem::forget` 和 unsafe 代码可以轻松绕过 |
| **脆弱性** | 插入一行看似无关的代码就可能让整个程序崩溃 |

---

## 相关资源

- [Jack Wrenn 的原文章：Undroppable Types in Rust](https://jack.wrenn.fyi/blog/undroppable-types-in-rust/)
- [prevent_drop crate](https://crates.io/crates/prevent_drop)
- [Rust 内部论坛关于 linear types 的讨论](https://internals.rust-lang.org/)
- [线性类型 - Wikipedia](https://en.wikipedia.org/wiki/Substructural_type_system#Linear_type_systems)
- [Rust 的 Affine Type System](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)

---

## 总结

这篇文章展示了一个**技术上有趣但实用性有限**的 Rust 技巧：

1. **核心思想**：利用 `const panic` 在编译时检测"未被使用的值"
2. **关键技巧**：`ManuallyDrop` 阻止 drop 代码生成 + `MaybeUninit` 零开销存储
3. **致命缺陷**：依赖编译器的静态分析能力，容易被无关代码干扰
4. **深层问题**：Rust 的类型系统设计上不支持真正的线性类型，这种实现只是"模拟"

这更像是一个**编程语言探索实验**，展示了 Rust 类型系统和编译器的边界，而不是推荐在生产代码中使用的技术。