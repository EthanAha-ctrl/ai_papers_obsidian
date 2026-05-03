











这个网站是 **PyO3** 项目的官方文档（版本 **0.28.3**），它是一个 **Rust** 与 **Python** 的互操作工具集，提供了 **双向绑定（bidirectional binding）** 能力：既可以用 **Rust** 编写高性能的 **Python C-extension** 模块，也可以将 **Python 解释器嵌入到 Rust 程序中** 直接调用 Python 代码。其核心基于 **Python/C API** 的 **FFI（Foreign Function Interface）** 封装。

---

## 1. 核心功能概览

PyO3 支持两种主要使用模式：

1. **Writing Rust-based Python modules**  
   - 使用 `#[pyclass]`、`#[pymethods]`、`#[pymodule]` 等 **attribute macro** 将 Rust 结构体/函数暴露给 Python。
   - 编译后生成 `.so` (Linux/macOS) 或 `.pyd` (Windows) 动态库，可直接通过 `pip install` 或 `python setup.py` 安装。

2. **Embedding Python in Rust binary**  
   - 在 Rust 中使用 `Python::new()` 或 `GILGuard` 创建 Python 解释器实例，通过 `PyModule::import()` 加载 Python 模块并调用其函数。
   - 适用于需要利用 Python 生态（如 NumPy、SciPy）的计算密集型 Rust 应用。

---

## 2. 技术架构深度解析

### 2.1 分层架构

PyO3 的架构分为三层，如官方 [Architecture.md](https://github.com/PyO3/pyo3/blob/main/Architecture.md) 所述：

```
┌─────────────────────────────────────┐
│      Safe Rust API (pyo3 crate)      │ ← 用户主要交互层
├─────────────────────────────────────┤
│   pyo3-ffi (low-level bindings)     │ ← 封装 Python.h API
├─────────────────────────────────────┤
│   Python/C API (CPython internals)  │ ← 底层 C 接口
└─────────────────────────────────────┘
```

- **Safe Rust API**：提供类型安全的包装器（如 `PyObject`、`Py<T>`、`Bound<'py, T>`），利用 **Rust 的生命周期（lifetime）** 和 **所有权（ownership）** 系统防止内存错误。
- **pyo3-ffi**：自动生成对 `Python.h` 中所有函数的 `extern "C"` 声明（通过 `bindgen`），确保与不同 Python 版本的 ABI 兼容。
- **Python/C API**：CPython 解释器的原生接口，例如 `PyObject_GetAttr`、`PyTuple_Pack` 等。PyO3 的 unsafe 代码块直接调用这些函数，而安全层封装了引用计数（reference counting）和错误处理。

### 2.2 关键类型与内存管理

#### 2.2.1 GIL（Global Interpreter Lock）
Python 使用 **GIL** 来保证内存管理的线程安全。任何线程在操作 Python 对象前**必须**获取 GIL。PyO3 通过 `Python<'py>` token 表示当前线程持有 GIL，其生命周期 `'py` 绑定到 GIL 的持有期。

#### 2.2.2 `Bound<'py, T>` vs `Py<T>`
- `Py<T>`：一个 **GIL-independent** 的引用，可以在没有持有 GIL 的情况下传递，但**不能**直接访问其字段或调用方法。它只是一个“拥有” Python 对象所有权的句柄（增加引用计数），类似于 `PyObject*`。
- `Bound<'py, T>`：一个 **GIL-bound** 的引用，生命周期 `'py` 证明当前线程持有 GIL。它可以安全地调用 Python 方法、访问属性。可视为 `(Py<T>, Python<'py>)` 的组合。

**示例代码片段**（来自 [Bound documentation](https://pyo3.rs/main/doc/pyo3/struct.Bound)）：

```rust
use pyo3::prelude::*;

#[pyclass]
struct MyClass {
    value: i32,
}

#[pymethods]
impl MyClass {
    #[new]
    fn new(value: i32) -> Self {
        Self { value }
    }

    fn get_value(&self) -> i32 {
        self.value
    }
}

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        // 在 GIL 作用域内创建 MyClass 实例
        let my_obj = MyClass::new(42).into_py(py);
        // 将 Py<MyClass> 转换为 Bound<'py, MyClass> 以调用方法
        let bound_obj = my_obj.bind(py);
        println!("Value: {}", bound_obj.get_value()?);
        Ok(())
    })
}
```

#### 2.2.3 引用计数与 `Drop` 实现
Python 对象使用 **引用计数** 管理生命周期。PyO3 的 `Py<T>` 在 `drop()` 时自动调用 `Py_DECREF`。`Bound<'py, T>` 不负责释放，因为它只借用 GIL 期间的使用权。

#### 2.2.4 `pyo3-ffi` 的 unsafe 封装
例如，`PyLong_AsLong` 的 Rust 绑定可能如下：

```rust
#[inline]
pub unsafe fn PyLong_AsLong(obj: *mut PyObject) -> libc::c_long {
    unsafe { (ffi::PyLong_AsLong)(obj) }
}
```

调用处必须确保 `obj` 是有效的 `PyLong` 类型，否则可能触发未定义行为。

---

## 3. 版本 0.28.3 的关键更新

根据 [CHANGELOG](https://pyo3.rs/v0.28.3/changelog) 和 [Releases](https://github.com/PyO3/pyo3/releases)，该版本主要是一个 **patch release**，包含：

- **Soundness fix for subclassing native types with `abi3`**：当启用 `abi3` feature 并设置最低 Python 版本时，使用 `#[pyclass(extends=PyList)]` 等继承原生 Python 类型的行为存在潜在安全漏洞，此版本已修复。
- 最小 Rust 版本要求 **1.83**。
- 支持 Python 3.8–3.12 及 PyPy 3.9–3.12。

**`abi3` feature 说明**：  
启用后，编译的扩展模块将使用 **有限 ABI（Limited API）**，确保在多个 Python 3.x 版本间兼容。但会限制对 CPython 内部结构的直接访问，因此某些高级功能可能受限。

---

## 4. 使用流程与编译依赖

### 4.1 `Cargo.toml` 配置

```toml
[lib]
name = "my_module"
crate-type = ["cdylib"]

[dependencies.pyo3]
version = "0.28.3"
# 可选 features：
# - "extension-module"：生成 Python 模块（默认不启用）
# - "auto-initialize"：自动调用 `pyo3::prepare_freethreaded_python()`（用于 embedding）
# - "abi3"：生成abi3兼容库
```

### 4.2 典型的 Python 模块示例

```rust
use pyo3::prelude::*;

#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[pymodule]
fn my_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

编译命令：`maturin build` 或 `cargo build --release`。

---

## 5. 与 Python/C API 的直接对比

| 操作 | Python/C API | PyO3 |
|------|--------------|------|
| 创建一个 Python int 对象 | `PyLong_FromLong(42)` | `Py::new(42).unwrap()` |
| 获取对象属性 | `PyObject_GetAttrString(obj, "name")` | `obj.getattr(py, "name")?` |
| 调用函数 | `PyObject_CallObject(func, args)` | `func.call(args, None)?` |
| 异常处理 | 返回 `NULL` 并检查 `PyErr_Occurred()` | 返回 `PyResult<T>`，`?` 自动传播异常 |

PyO3 将 **手动引用计数、错误检查** 转化为 **Rust 的所有权语义和 `Result` 类型**，大幅降低出错概率。

---

## 6. 性能考量

- **Zero-cost abstraction**：PyO3 的安全包装在 Release 模式下通常被优化掉，性能接近直接调用 Python/C API。
- **GIL 释放**：通过 `py.allow_threads(|| { /* 长时间计算 */ })` 可以在执行 Rust 代码时释放 GIL，提升并发性能。
- **No GIL 共享**：在 embedding 模式下，可使用 `Py::new()` 创建不依赖 GIL 的对象，但操作时仍需重新获取 GIL。

---

## 7. 相关资源链接

- 官方文档：[https://pyo3.rs/v0.28.3/](https://pyo3.rs/v0.28.3/)
- GitHub 仓库：[https://github.com/pyo3/pyo3](https://github.com/pyo3/pyo3)
- API 详细文档（Docs.rs）：[https://docs.rs/pyo3/latest/pyo3/](https://docs.rs/pyo3/latest/pyo3/)
- 架构设计文档：[https://github.com/PyO3/pyo3/blob/main/Architecture.md](https://github.com/PyO3/pyo3/blob/main/Architecture.md)
- 内存管理深入分析：[https://gist.github.com/benkay86/957cf17d2ce2bab3ec47fb92320e75e4](https://gist.github.com/benkay86/957cf17d2ce2bab3ec47fb92320e75e4)

---

综上所述，**https://pyo3.rs/v0.28.3/** 是 PyO3 项目针对 0.28.3 版本的文档网站，它系统性地介绍了如何用 Rust 与 Python 进行高性能、内存安全的互操作，涵盖了从基础使用、类型系统设计到底层 FFI 实现的全部技术细节。