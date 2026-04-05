







## 什么是 Prolog？——深入理解逻辑编程语言

---

### 一、基本定义与核心理念

**Prolog** (PROgrammation en LOGique，法语"逻辑编程"的缩写) 是一种**逻辑编程语言**（logic programming language），诞生于 **1972年**，由法国艾克斯-马赛大学的 **Alain Colmerauer** 和 **Philippe Roussel** 共同创造。

Prolog 的核心理念是：

> **程序 = 事实 + 规则**
> **计算 = 查询**

与传统命令式语言不同，Prolog 是**声明式**的——你描述"是什么"（what），而不是"怎么做"（how）。程序逻辑通过**关系**（relations）表达，计算通过**查询**（query）触发。

---

### 二、理论基础：一阶逻辑与 Horn 子句

#### 2.1 一阶谓词逻辑

Prolog 建立在**一阶逻辑**（First-Order Logic, FOL）基础上。一阶逻辑包含：

| 元素 | 符号 | 含义 |
|------|------|------|
| 常量 | a, b, c | 特定个体 |
| 变量 | X, Y, Z | 可替换的占位符 |
| 函数符号 | f(t₁,...,tₙ) | 复合项 |
| 谓词符号 | P(t₁,...,tₙ) | 关系或属性 |
| 量词 | ∀, ∃ | 全称/存在量化 |

#### 2.2 Horn 子句

Prolog 的纯子集基于 **Horn 子句**（Horn Clause）。Horn 子句是**最多只有一个正文字**的析取子句：

**一般形式：**
```
H :- B₁, B₂, ..., Bₙ
```

**逻辑等价：**
$$H \leftarrow B_1 \land B_2 \land ... \land B_n$$

或写作：
$$H \lor \neg B_1 \lor \neg B_2 \lor ... \lor \neg B_n$$

**Horn 子句的两种类型：**

| 类型 | 形式 | Prolog 中对应 |
|------|------|---------------|
| 事实 | `H.` | 无主体的规则 |
| 规则 | `H :- B.` | 有主体的规则 |

**Prolog 的过程性解释：**
```
要证明 H，需要证明 B₁ 且 B₂ 且 ... 且 Bₙ
```

---

### 三、数据类型与语法结构

Prolog 只有**一种数据类型**：**项**。

#### 3.1 项的四种形式

```
Term ::= Atom | Number | Variable | CompoundTerm
```

| 类型 | 定义 | 示例 |
|------|------|------|
| **原子** | 小写字母开头或单引号包围 | `x`, `red`, `'Hello World'` |
| **数字** | 整数或浮点数 | `42`, `3.14` |
| **变量** | 大写字母或下划线开头 | `X`, `_Var`, `_`（匿名变量） |
| **复合项** | 函子 + 参数列表 | `person(name, age)` |

#### 3.2 特殊复合项

**列表：**
```prolog
[1, 2, 3]      % 糖衣语法
.(1, .(2, .(3, [])))  % 实际结构
[H | T]        % 头尾分解，H=1, T=[2,3]
```

**字符串：**
```prolog
"hello"  % 等价于 [104, 101, 108, 108, 111]（ASCII码列表）
```

---

### 四、程序结构：事实、规则与查询

#### 4.1 事实

事实是**无条件为真**的陈述：

```prolog
human(socrates).
father_child(tom, sally).
mother_child(trude, sally).
```

等价于：
```prolog
human(socrates) :- true.   % true 是永真的内置谓词
```

#### 4.2 规则

规则定义**条件推导**：

```prolog
mortal(X) :- human(X).                    % 凡人即人类
sibling(X, Y) :- parent_child(Z, X),
                 parent_child(Z, Y),
                 X \= Y.                  % 兄弟姐妹 = 同父同母且不同人
parent_child(X, Y) :- father_child(X, Y). % 父亲关系
parent_child(X, Y) :- mother_child(X, Y). % 母亲关系
```

#### 4.3 查询

查询是向 Prolog 引擎提出的问题：

```prolog
?- human(socrates).
   Yes                              % Socrates 是人类吗？

?- human(X).
   X = socrates                     % 谁是人类？

?- mortal(X).
   X = socrates                     % 通过规则推导：谁必死？

?- sibling(sally, erica).
   Yes                              % Sally 和 Erica 是兄弟姐妹吗？
```

---

### 五、执行机制：SLD Resolution 与 Backtracking

#### 5.1 SLD Resolution

Prolog 使用 **SLD Resolution**（Selective Linear Definite clause resolution）作为推理引擎。

**核心思想：**
给定查询 Q 和程序 P，Prolog 尝试找到**否定查询的反驳**（resolution refutation of the negated query）：

$$P \cup \{\neg Q\} \vdash \bot$$

如果能证明矛盾，则原查询 Q 成立。

#### 5.2 执行流程详解

以查询 `?- sibling(sally, erica).` 为例：

```
程序：
  mother_child(trude, sally).
  father_child(tom, sally).
  father_child(tom, erica).
  sibling(X, Y) :- parent_child(Z, X), parent_child(Z, Y), X \= Y.
  parent_child(X, Y) :- father_child(X, Y).
  parent_child(X, Y) :- mother_child(X, Y).
```

**执行树：**

```
sibling(sally, erica)
    |
    ├─ 展开 sibling 规则
    │   parent_child(Z, sally), parent_child(Z, erica), sally \= erica
    │
    ├─ parent_child(Z, sally)
    │   ├─ 选择 father_child(Z, sally) → Z = tom ✓
    │   │   └─ parent_child(tom, erica)
    │   │       └─ father_child(tom, erica) ✓
    │   │           └─ sally \= erica ✓
    │   │               └─ 成功！
    │   │
    │   └─ 若失败，回溯尝试 mother_child(Z, sally)
    │
    └─ 若全部失败，查询失败
```

#### 5.3 Choice Point 与 Backtracking

当多个子句头匹配同一目标时，Prolog 创建**选择点**：

```
目标: parent_child(Z, sally)

匹配的子句：
  parent_child(X, Y) :- father_child(X, Y).    % 选择点 #1
  parent_child(X, Y) :- mother_child(X, Y).    % 选择点 #2
```

**回溯机制：**
1. 首先尝试第一个选择
2. 如果后续目标失败，**撤销所有变量绑定**
3. 尝试下一个选择
4. 重复直到成功或穷尽所有选择

#### 5.4 Unification（合一）

Unification 是 Prolog 的核心匹配算法：

**定义：** 找到最一般合一者，使两个项相等。

**算法：**

```prolog
unify(X, X) :- !.                          % 相同项直接成功
unify(X, Y) :- var(X), !, X = Y.           % 变量绑定到项
unify(X, Y) :- var(Y), !, Y = X.           % 同上
unify(F1-Args1, F2-Args2) :-               % 复合项
    F1 = F2,                               % 函子相同
    same_length(Args1, Args2),             % 参数数量相同
    maplist(unify, Args1, Args2).          % 递归合一参数
```

**示例：**

```prolog
?- person(X, 25) = person(john, Y).
   X = john, Y = 25.                        % 双向绑定

?- [H|T] = [1,2,3].
   H = 1, T = [2,3].

?- f(X, X) = f(a, b).
   false.                                   % X 不能同时等于 a 和 b
```

---

### 六、高级特性

#### 6.1 Cut 操作符 (`!`)

Cut 是 Prolog 的**控制流操作符**，用于剪枝：

```prolog
max(X, Y, X) :- X >= Y, !.     % 如果 X >= Y，不再尝试下一规则
max(X, Y, Y) :- X < Y.

% 更紧凑的写法
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
```

**Cut 的效果：**
- 停止对当前谓词剩余子句的尝试
- 冻结当前选择点
- 防止回溯到当前谓词调用点之前

**问题：** Cut 破坏了纯声明式特性，需要**过程性阅读**程序。

#### 6.2 Negation as Failure

```prolog
legal(X) :- \+ illegal(X).     % X 是合法的，如果不能证明 X 是非法的
```

**语义：**
- `\+ Goal` 尝试证明 Goal
- 如果成功，`\+ Goal` 失败
- 如果失败（无法证明），`\+ Goal` 成功

**注意：** 这是一种**非单调推理**（non-monotonic reasoning），仅在参数为**基项**（ground term，无变量）时是可靠的。

#### 6.3 Definite Clause Grammars (DCG)

DCG 是 Prolog 的**语法扩展**，用于自然语言处理：

```prolog
% 语法规则
sentence --> noun_phrase, verb_phrase.
noun_phrase --> determiner, noun.
verb_phrase --> verb, noun_phrase.

% 词汇表
determiner --> [the].
noun --> [cat]; [dog]; [mouse].
verb --> [chases]; [eats].

% 查询
?- sentence([the, cat, chases, the, mouse], []).
   true.
```

**展开机制：**
```prolog
% DCG 规则
sentence --> noun_phrase, verb_phrase.

% 展开为普通 Prolog 子句
sentence(S0, S) :-
    noun_phrase(S0, S1),
    verb_phrase(S1, S).
```

两个隐式参数 `S0` 和 `S` 代表**输入字符串的剩余部分**，实现了**差异列表**（difference list）的高效处理。

---

### 七、实现技术

#### 7.1 Warren Abstract Machine (WAM)

WAM 是 Prolog 的**标准抽象机器**，由 David H.D. Warren 设计。

**WAM 架构：**

```
┌────────────────────────────────────────────┐
│                 WAM                        │
├────────────────────────────────────────────┤
│                                            │
│   ┌──────────┐    ┌──────────────────┐    │
│   │  Code    │    │   Data Areas      │    │
│   │  Area    │    ├──────────────────┤    │
│   └──────────┘    │ Stack (Local)   │    │
│                   │   - Environments │    │
│   ┌──────────┐    │   - Choice pts   │    │
│   │ Registers│    ├──────────────────┤    │
│   │  S, P, CP│    │ Heap (Global)    │    │
│   │  H, HB...│    │   - Terms        │    │
│   └──────────┘    ├──────────────────┤    │
│                   │ Trail            │    │
│                   │   - Undo bindings│    │
│                   └──────────────────┘    │
│                                            │
└────────────────────────────────────────────┘
```

**关键寄存器：**

| 寄存器 | 用途 |
|--------|------|
| `S` | 结构指针，当前处理项 |
| `P` | 程序指针，下一条指令 |
| `CP` | 继续指针，返回地址 |
| `H` | 堆顶指针 |
| `HB` | 堆回溯点 |
| `B` | 选择点栈顶 |
| `TR` | Trail 栈顶 |

#### 7.2 Term Indexing

直接搜索可合一子句是 **O(n)** 复杂度。Term Indexing 实现了**亚线性查找**：

```prolog
% 没有索引：必须检查所有子句
foo(a, ...).
foo(b, ...).
foo(c, ...).
...

% 有索引：根据第一个参数直接定位
foo(a, ...) :- ...   % 哈希表：a -> [子句1]
foo(b, ...) :- ...   % 哈希表：b -> [子句2]
```

**索引技术：**
- **主键索引：** 仅第一个参数
- **多参数索引：** 使用 **superimposed codewords** 或 **field-encoded words**
- **JIT 索引：** 运行时动态创建

#### 7.3 Tabling (Memoization)

**问题：** 左递归会导致无限循环：

```prolog
path(X, Y) :- path(X, Z), edge(Z, Y).
path(X, Y) :- edge(X, Y).
```

**解决方案：Tabling**

```prolog
:- table path/2.    % 声明 path 使用 tabling

path(X, Y) :- path(X, Z), edge(Z, Y).
path(X, Y) :- edge(X, Y).
```

**原理：**
```
┌─────────────────────────────────────┐
│          Tabled Evaluation          │
├─────────────────────────────────────┤
│  1. 遇到新的子目标 → 创建表条目      │
│  2. 子目标再次出现 → 复用已有答案    │
│  3. 避免重复计算，终止性保证          │
└─────────────────────────────────────┘
```

支持 **SLG Resolution** 或 **Linear Tabling**。

---

### 八、代码示例

#### 8.1 QuickSort

```prolog
quicksort([], []).
quicksort([Pivot|Rest], Sorted) :-
    partition(Pivot, Rest, Smaller, Larger),
    quicksort(Smaller, SortedSmaller),
    quicksort(Larger, SortedLarger),
    append(SortedSmaller, [Pivot|SortedLarger], Sorted).

partition(_, [], [], []).
partition(Pivot, [X|Xs], [X|Smaller], Larger) :-
    X @< Pivot, !,
    partition(Pivot, Xs, Smaller, Larger).
partition(Pivot, [X|Xs], Smaller, [X|Larger]) :-
    partition(Pivot, Xs, Smaller, Larger).
```

#### 8.2 元解释器

```prolog
% 最简元解释器
solve(true).
solve((A, B)) :-
    solve(A),
    solve(B).
solve(Goal) :-
    clause(Goal, Body),
    solve(Body).

% 带置信度的元解释器
solve(true, 1.0) :- !.
solve((A, B), Cert) :-
    solve(A, CertA),
    solve(B, CertB),
    Cert is min(CertA, CertB).
solve(Goal, Cert) :-
    clause_cf(Goal, Body, RuleCert),
    solve(Body, BodyCert),
    Cert is RuleCert * BodyCert.
```

#### 8.3 Turing Machine 模拟

```prolog
turing(Tape0, Tape) :-
    perform(q0, [], Ls, Tape0, Rs),
    reverse(Ls, Ls1),
    append(Ls1, Rs, Tape).

perform(qf, Ls, Ls, Rs, Rs) :- !.
perform(Q0, Ls0, Ls, Rs0, Rs) :-
    symbol(Rs0, Sym, RsRest),
    once(rule(Q0, Sym, Q1, NewSym, Action)),
    action(Action, Ls0, Ls1, [NewSym|RsRest], Rs1),
    perform(Q1, Ls1, Ls, Rs1, Rs).

% 示例机器：一元数加一
rule(q0, 1, q0, 1, right).
rule(q0, b, qf, 1, stay).

?- turing([1,1,1], Ts).
   Ts = [1, 1, 1, 1].
```

---

### 九、应用领域

| 领域 | 应用 |
|------|------|
| **自然语言处理** | 语法分析、语义解析（原始设计目标） |
| **专家系统** | 知识库推理、诊断系统 |
| **定理证明** | 自动推理、类型系统验证 |
| **规划系统** | 自动规划、调度 |
| **数据库查询** | Datalog 关系查询 |
| **IBM Watson** | 自然语言解析树的 pattern matching |
| **图数据库** | TerminusDB 知识图谱 |

**IBM Watson 中的应用：**
> "We required a language in which we could conveniently express pattern matching rules over the parse trees and other annotations... We found that Prolog was the ideal choice due to its simplicity and expressiveness."

---

### 十、与其他语言的关系

```
                    Planner (1970s)
                        │
                        ▼
         ┌──────────────────────────────┐
         │         Prolog (1972)         │
         └──────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬───────────────┐
        ▼               ▼               ▼               ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Erlang  │   │ Datalog  │   │ Mercury  │   │  CHR     │
   │(并发)   │   │(子集)    │   │(类型化)  │   │(约束)    │
   └─────────┘   └──────────┘   └──────────┘   └──────────┘
        │               │
        ▼               ▼
   ┌─────────┐   ┌──────────┐
   │Clojure  │   │  Oz/Mozart│
   └─────────┘   └──────────┘
```

**关键衍生语言：**

- **Datalog：** Prolog 子集，非 Turing-complete，完全声明式
- **Erlang：** 原基于 Prolog 实现，保留 unification-based 语法
- **Mercury：** 强类型 + 模式系统
- **λProlog：** 高阶逻辑 + 多态类型
- **Visual Prolog：** 强类型 + 面向对象

---

### 十一、Prolog 的哲学思考

#### 11.1 声明式 vs 命令式

```
┌─────────────────────────────────────────────────────────┐
│                    传统命令式                            │
│  "怎么做"：                                              │
│    for (i=0; i<n; i++) {                                │
│        if (condition) do_something();                   │
│    }                                                    │
├─────────────────────────────────────────────────────────┤
│                    Prolog 声明式                         │
│  "是什么"：                                              │
│    result(X) :- condition(X), action(X).                │
│    % 让引擎找出所有满足条件的 X                           │
└─────────────────────────────────────────────────────────┘
```

#### 11.2 双向性

Prolog 程序可以**双向运行**：

```prolog
append([], L, L).
append([H|T], L, [H|R]) :- append(T, L, R).

% 正向：拼接列表
?- append([1,2], [3,4], X).
   X = [1,2,3,4].

% 反向：拆分列表
?- append(X, Y, [1,2,3,4]).
   X = [], Y = [1,2,3,4] ;
   X = [1], Y = [2,3,4] ;
   X = [1,2], Y = [3,4] ;
   ...

% 检查：是否可以拼接
?- append([1,2], [3], [1,2,3]).
   true.
```

#### 11.3 Homoiconicity（同像性）

Prolog 程序本身就是数据结构：

```prolog
?- clause(append(A,B,C), Body).
   A = [], B = C, Body = true ;
   A = [H|T], C = [H|R], Body = append(T, B, R).
```

这使得**元编程**极为自然。

---

### 十二、资源与参考

**参考链接：**

- ISO Prolog 标准：https://www.iso.org/standard/21413.html
- SWI-Prolog（最流行的开源实现）：https://www.swi-prolog.org/
- Learn Prolog Now!：http://www.learnprolognow.org/
- Prolog: Programming for Artificial Intelligence (Ivan Bratko)：经典教材

**历史论文：**
- Colmerauer & Roussel, "The Birth of Prolog" (1996)：http://www.alain-colmerauer.info/
- Kowalski, "The Early Years of Logic Programming" (1988)

---

**总结：** Prolog 是一种将**数理逻辑**直接转化为**可执行程序**的语言。它用 Horn 子句描述世界，用 SLD Resolution 自动推理，用 Backtracking 探索所有可能性。这种范式特别适合需要**符号推理**、**知识表示**和**关系查询**的问题。虽然在工业界应用有限，但在 AI、NLP 和专家系统领域仍有独特价值。